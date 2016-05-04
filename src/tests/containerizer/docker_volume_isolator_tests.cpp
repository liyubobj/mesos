// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <mesos/mesos.hpp>

#include <process/gtest.hpp>

#include <stout/os.hpp>
#include <stout/path.hpp>

#include "linux/fs.hpp"

#include "slave/containerizer/mesos/containerizer.hpp"
#include "slave/containerizer/mesos/linux_launcher.hpp"

#include "slave/containerizer/mesos/isolators/filesystem/linux.hpp"
#include "slave/containerizer/mesos/isolators/docker/volume/isolator.hpp"
#include "slave/containerizer/mesos/isolators/docker/volume/driver.hpp"

#include "tests/flags.hpp"
#include "tests/mesos.hpp"

namespace slave = mesos::internal::slave;

using process::Future;
using process::Owned;

using std::string;
using std::vector;

using mesos::internal::master::Master;

using mesos::internal::slave::Fetcher;
using mesos::internal::slave::Launcher;
using mesos::internal::slave::DockerVolumeIsolatorProcess;
using mesos::internal::slave::LinuxFilesystemIsolatorProcess;
using mesos::internal::slave::LinuxLauncher;
using mesos::internal::slave::MesosContainerizer;
using mesos::internal::slave::MesosContainerizerProcess;
using mesos::internal::slave::Provisioner;
using mesos::internal::slave::Slave;

using mesos::master::detector::MasterDetector;

using mesos::slave::ContainerLogger;
using mesos::slave::Isolator;

using slave::docker::volume::DriverClient;

using testing::DoAll;

namespace mesos {
namespace internal {
namespace tests {

class MockDockerVolumeDriverClient : public DriverClient
{
public:
  MockDockerVolumeDriverClient() {}

  virtual ~MockDockerVolumeDriverClient() {}

  MOCK_METHOD3(
      mount,
      Future<string>(
          const string& driver,
          const string& name,
          const hashmap<string, string>& options));

  MOCK_METHOD2(
      unmount,
      Future<Nothing>(
          const string& driver,
          const string& name));
};


class DockerVolumeIsolatorTest : public MesosTest
{
protected:
  virtual void TearDown()
  {
    // Try to remove any mounts under sandbox.
    if (::geteuid() == 0) {
      Try<Nothing> unmountAll = fs::unmountAll(sandbox->c_str(), MNT_DETACH);
      if (unmountAll.isError()) {
        LOG(ERROR) << "Failed to unmount '" << sandbox->c_str()
                   << "': " << unmountAll.error();
        return;
      }
    }

    MesosTest::TearDown();
  }

  Volume createDockerVolume(
      const string& driver,
      const string& name,
      const string& containerPath,
      const Option<hashmap<string, string>>& options = None())
  {
    Volume volume;
    volume.set_mode(Volume::RW);
    volume.set_container_path(containerPath);

    Volume::Source* source = volume.mutable_source();
    source->set_type(Volume::Source::DOCKER_VOLUME);

    Volume::Source::DockerVolume* docker = source->mutable_docker_volume();
    docker->set_driver(driver);
    docker->set_name(name);

    Parameters parameters;

    if (options.isSome()) {
      foreachpair (const string& key, const string& value, options.get()) {
        Parameter* parameter = parameters.add_parameter();
        parameter->set_key(key);
        parameter->set_value(value);
      }

      docker->mutable_driver_options()->CopyFrom(parameters);
    }

    return volume;
  }

  // This helper creates a MesosContainerizer instance that uses the
  // LinuxFilesystemIsolator and DockerVolumeIsolator.
  Try<Owned<MesosContainerizer>> createContainerizer(
      const slave::Flags& flags,
      const Owned<DriverClient>& mockClient)
  {
    Try<Isolator*> linuxIsolator =
      LinuxFilesystemIsolatorProcess::create(flags);

    if (linuxIsolator.isError()) {
      return Error(
          "Failed to create LinuxFilesystemIsolator: " +
          linuxIsolator.error());
    }

    Try<Isolator*> volumeIsolator =
      DockerVolumeIsolatorProcess::_create(flags, mockClient);

    if (volumeIsolator.isError()) {
      return Error(
          "Failed to create DockerVolumeIsolator: " +
          volumeIsolator.error());
    }

    Try<Launcher*> launcher = LinuxLauncher::create(flags);
    if (launcher.isError()) {
      return Error("Failed to create LinuxLauncher: " + launcher.error());
    }

    // Create and initialize a new container logger.
    Try<ContainerLogger*> logger =
      ContainerLogger::create(flags.container_logger);

    if (logger.isError()) {
      return Error("Failed to create container logger: " + logger.error());
    }

    Try<Owned<Provisioner>> provisioner = Provisioner::create(flags);
    if (provisioner.isError()) {
      return Error("Failed to create provisioner: " + provisioner.error());
    }

    return Owned<MesosContainerizer>(
        new MesosContainerizer(
            flags,
            false,
            &fetcher,
            Owned<ContainerLogger>(logger.get()),
            Owned<Launcher>(launcher.get()),
            provisioner.get(),
            {Owned<Isolator>(linuxIsolator.get()),
             Owned<Isolator>(volumeIsolator.get())}));
  }

private:
  Fetcher fetcher;
};


// This test verifies that multiple docker volumes with both absolute
// path and relative path are properly mounted to a container without
// rootfs, and launches a command task that reads files from the
// mounted docker volumes.
TEST_F(DockerVolumeIsolatorTest, ROOT_CommandTaskNoRootfsWithVolumes)
{
  Try<Owned<cluster::Master>> master = StartMaster();
  ASSERT_SOME(master);

  slave::Flags flags = CreateSlaveFlags();

  MockDockerVolumeDriverClient* mockClient =
      new MockDockerVolumeDriverClient;

  Try<Owned<MesosContainerizer>> containerizer =
    createContainerizer(flags, Owned<DriverClient>(mockClient));

  ASSERT_SOME(containerizer);

  Owned<MasterDetector> detector = master.get()->createDetector();

  Try<Owned<cluster::Slave>> slave = StartSlave(
      detector.get(),
      containerizer.get().get(),
      flags);

  ASSERT_SOME(slave);

  MockScheduler sched;

  MesosSchedulerDriver driver(
      &sched,
      DEFAULT_FRAMEWORK_INFO,
      master.get()->pid,
      DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  Future<vector<Offer>> offers;
  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return()); // Ignore subsequent offers.

  driver.start();

  AWAIT_READY(frameworkId);

  AWAIT_READY(offers);
  ASSERT_NE(0u, offers->size());

  const Offer& offer = offers.get()[0];

  const string key = "iops";
  const string value = "150";

  hashmap<string, string> options = {{key, value}};

  // Create a volume with relative path.
  const string driver1 = "driver1";
  const string name1 = "name1";
  const string containerPath1 = "tmp/foo1";

  Volume volume1 = createDockerVolume(driver1, name1, containerPath1, options);

  // Create a volume with absolute path.
  const string driver2 = "driver2";
  const string name2 = "name2";

  // Make sure the absolute path exist.
  const string containerPath2 = path::join(os::getcwd(), "foo2");
  ASSERT_SOME(os::mkdir(containerPath2));

  Volume volume2 = createDockerVolume(driver2, name2, containerPath2);

  TaskInfo task = createTask(
      offer.slave_id(),
      offer.resources(),
      "test -f " + containerPath1 + "/file1 && "
      "test -f " + containerPath2 + "/file2;");

  ContainerInfo containerInfo;
  containerInfo.set_type(ContainerInfo::MESOS);
  containerInfo.add_volumes()->CopyFrom(volume1);
  containerInfo.add_volumes()->CopyFrom(volume2);

  task.mutable_container()->CopyFrom(containerInfo);

  // Create mount point for volume1.
  const string mountPoint1 = path::join(os::getcwd(), "volume1");
  ASSERT_SOME(os::mkdir(mountPoint1));
  ASSERT_SOME(os::touch(path::join(mountPoint1, "file1")));

  // Create mount point for volume2.
  const string mountPoint2 = path::join(os::getcwd(), "volume2");
  ASSERT_SOME(os::mkdir(mountPoint2));
  ASSERT_SOME(os::touch(path::join(mountPoint2, "file2")));

  Future<string> mount1Name;
  Future<string> mount2Name;
  Future<hashmap<string, string>> mount1Options;

  EXPECT_CALL(*mockClient, mount(driver1, _, _))
    .WillOnce(DoAll(FutureArg<1>(&mount1Name),
                    FutureArg<2>(&mount1Options),
                    Return(mountPoint1)));

  EXPECT_CALL(*mockClient, mount(driver2, _, _))
    .WillOnce(DoAll(FutureArg<1>(&mount2Name),
                    Return(mountPoint2)));

  Future<string> unmount1Name;
  Future<string> unmount2Name;

  EXPECT_CALL(*mockClient, unmount(driver1, _))
    .WillOnce(DoAll(FutureArg<1>(&unmount1Name),
                    Return(Nothing())));

  EXPECT_CALL(*mockClient, unmount(driver2, _))
    .WillOnce(DoAll(FutureArg<1>(&unmount2Name),
                    Return(Nothing())));

  Future<TaskStatus> statusRunning;
  Future<TaskStatus> statusFinished;

  EXPECT_CALL(sched, statusUpdate(&driver, _))
    .WillOnce(FutureArg<1>(&statusRunning))
    .WillOnce(FutureArg<1>(&statusFinished));

  driver.launchTasks(offer.id(), {task});

  AWAIT_READY(statusRunning);
  EXPECT_EQ(TASK_RUNNING, statusRunning->state());

  // Make sure the docker volume mount parameters are same with the
  // parameters in `containerInfo`.
  AWAIT_EXPECT_EQ(name1, mount1Name);
  AWAIT_EXPECT_EQ(name2, mount2Name);

  AWAIT_READY(mount1Options);
  EXPECT_SOME_EQ(value, mount1Options->get(key));

  AWAIT_READY(statusFinished);
  EXPECT_EQ(TASK_FINISHED, statusFinished->state());

  // Make sure the docker volume unmount parameters are same with
  // the parameters in `containerInfo`.
  AWAIT_EXPECT_EQ(name1, unmount1Name);
  AWAIT_EXPECT_EQ(name2, unmount2Name);

  driver.stop();
  driver.join();
}

} // namespace tests {
} // namespace internal {
} // namespace mesos {
