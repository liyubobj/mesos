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

#include <set>
#include <vector>

#include <gmock/gmock.h>

#include <mesos/resources.hpp>
#include <mesos/scheduler.hpp>

#include <mesos/master/detector.hpp>

#include <process/future.hpp>
#include <process/gtest.hpp>
#include <process/owned.hpp>

#include <stout/jsonify.hpp>

#include "master/master.hpp"

#include "slave/slave.hpp"

#include "slave/containerizer/containerizer.hpp"
#include "slave/containerizer/fetcher.hpp"

#include "slave/containerizer/mesos/isolators/gpu/nvidia.hpp"

#include "tests/flags.hpp"
#include "tests/mesos.hpp"

using mesos::internal::master::Master;

using mesos::internal::slave::Containerizer;
using mesos::internal::slave::Fetcher;
using mesos::internal::slave::Gpu;
using mesos::internal::slave::MesosContainerizer;
using mesos::internal::slave::MesosContainerizerProcess;
using mesos::internal::slave::NvidiaGpuAllocator;
using mesos::internal::slave::Slave;

using mesos::master::detector::MasterDetector;

using mesos::internal::slave::DockerContainerizer;
using mesos::internal::slave::DockerContainerizerProcess;
using mesos::slave::ContainerLogger;

using namespace process;

using std::set;
using std::vector;
using std::list;

using testing::_;
using testing::Eq;
using testing::Return;

namespace mesos {
namespace internal {
namespace tests {

class NvidiaGpuTest : public MesosTest {};
class NvidiaGpuAllocatorTest : public MesosTest {};
class NvidiaGpuDockerContainerizerTest: public MesosTest
{
public:
  static string containerName(
      const SlaveID& slaveId,
      const ContainerID& containerId)
  {
    return slave::DOCKER_NAME_PREFIX + slaveId.value() +
      slave::DOCKER_NAME_SEPERATOR + containerId.value();
  }

  enum ContainerState
  {
    EXISTS,
    RUNNING
  };

  static bool exists(
      const process::Shared<Docker>& docker,
      const SlaveID& slaveId,
      const ContainerID& containerId,
      ContainerState state = ContainerState::EXISTS)
  {
    Duration waited = Duration::zero();
    string expectedName = containerName(slaveId, containerId);

    do {
      Future<Docker::Container> inspect = docker->inspect(expectedName);

      if (!inspect.await(Seconds(3))) {
        return false;
      }

      if (inspect.isReady()) {
        switch (state) {
          case ContainerState::RUNNING:
            if (inspect.get().pid.isSome()) {
              return true;
            }
            // Retry looking for running pid until timeout.
            break;
          case ContainerState::EXISTS:
            return true;
        }
      }

      os::sleep(Milliseconds(200));
      waited += Milliseconds(200);
    } while (waited < Seconds(5));

    return false;
  }

  static bool containsLine(
    const vector<string>& lines,
    const string& expectedLine)
  {
    foreach (const string& line, lines) {
      if (line == expectedLine) {
        return true;
      }
    }

    return false;
  }

  virtual void TearDown()
  {
    Try<Owned<Docker>> docker = Docker::create(
        tests::flags.docker,
        tests::flags.docker_socket,
        false);

    ASSERT_SOME(docker);

    Future<list<Docker::Container>> containers =
      docker.get()->ps(true, slave::DOCKER_NAME_PREFIX);

    AWAIT_READY(containers);

    // Cleanup all mesos launched containers.
    foreach (const Docker::Container& container, containers.get()) {
      AWAIT_READY_FOR(docker.get()->rm(container.id, true), Seconds(30));
    }
  }

  Option<set<string>> parseInspectDevices(const string& inspect) {
    set<string> devices;
    Try<JSON::Array> parse = JSON::parse<JSON::Array>(inspect);
    if (parse.isError()) {
        return None();
    }

    JSON::Array array = parse.get();
    if (array.values.size() != 1) {
        return None();
    }

    CHECK(array.values.front().is<JSON::Object>());
    JSON::Object json = array.values.front().as<JSON::Object>();

    Option<set<string>> hostDevices = set<string>();
    Result<JSON::Array> deviceJson =
      json.find<JSON::Array>("HostConfig.Devices");
    if (deviceJson.isSome()) {
      // Get elements in the array and push it to devices set
      const vector<JSON::Value> values = deviceJson.get().values;
      if (values.size() != 0) {
        foreach(const JSON::Value& value, values) {
          if (value.is<JSON::Object>()) {
            Result<JSON::String> devicePath =
              value.as<JSON::Object>().find<JSON::String>("PathOnHost");
            if (devicePath.isSome()) {
              // Push the device to the set
              hostDevices.get().insert(devicePath.get().value);
            }
          }
        }
      }
    }
    return hostDevices;
  }
};

// This test verifies that we are able to enable the Nvidia GPU
// isolator and launch tasks with restricted access to GPUs. We
// first launch a task with access to 0 GPUs and verify that a
// call to `nvidia-smi` fails. We then launch a task with 1 GPU
// and verify that a call to `nvidia-smi` both succeeds and
// reports exactly 1 GPU available.
TEST_F(NvidiaGpuTest, ROOT_CGROUPS_NVIDIA_GPU_VerifyDeviceAccess)
{
  Try<Owned<cluster::Master>> master = StartMaster();
  ASSERT_SOME(master);

  // Turn on Nvidia GPU isolation.
  // Assume at least one GPU is available for isolation.
  slave::Flags flags = CreateSlaveFlags();
  flags.isolation = "cgroups/devices,gpu/nvidia";
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:1";

  Owned<MasterDetector> detector = master.get()->createDetector();

  Try<Owned<cluster::Slave>> slave = StartSlave(detector.get(), flags);
  ASSERT_SOME(slave);

  MockScheduler sched;

  FrameworkInfo frameworkInfo = DEFAULT_FRAMEWORK_INFO;
  frameworkInfo.add_capabilities()->set_type(
      FrameworkInfo::Capability::GPU_RESOURCES);

  MesosSchedulerDriver driver(
      &sched, frameworkInfo, master.get()->pid, DEFAULT_CREDENTIAL);

  Future<Nothing> schedRegistered;
  EXPECT_CALL(sched, registered(_, _, _))
    .WillOnce(FutureSatisfy(&schedRegistered));

  Future<vector<Offer>> offers1, offers2;
  EXPECT_CALL(sched, resourceOffers(_, _))
    .WillOnce(FutureArg<1>(&offers1))
    .WillOnce(FutureArg<1>(&offers2))
    .WillRepeatedly(Return());      // Ignore subsequent offers.

  driver.start();

  AWAIT_READY(schedRegistered);

  // Launch a task requesting no GPUs and
  // verify that running `nvidia-smi` fails.
  AWAIT_READY(offers1);
  EXPECT_EQ(1u, offers1->size());

  TaskInfo task1 = createTask(
      offers1.get()[0].slave_id(),
      Resources::parse("cpus:0.1;mem:128;").get(),
      "nvidia-smi");

  Future<TaskStatus> statusRunning1, statusFailed1;
  EXPECT_CALL(sched, statusUpdate(_, _))
    .WillOnce(FutureArg<1>(&statusRunning1))
    .WillOnce(FutureArg<1>(&statusFailed1));

  driver.launchTasks(offers1.get()[0].id(), {task1});

  AWAIT_READY(statusRunning1);
  ASSERT_EQ(TASK_RUNNING, statusRunning1->state());

  AWAIT_READY(statusFailed1);
  ASSERT_EQ(TASK_FAILED, statusFailed1->state());

  // Launch a task requesting 1 GPU and verify
  // that `nvidia-smi` lists exactly one GPU.
  AWAIT_READY(offers2);
  EXPECT_EQ(1u, offers2->size());

  TaskInfo task2 = createTask(
      offers1.get()[0].slave_id(),
      Resources::parse("cpus:0.1;mem:128;gpus:1").get(),
      "NUM_GPUS=`nvidia-smi --list-gpus | wc -l`;\n"
      "if [ \"$NUM_GPUS\" != \"1\" ]; then\n"
      "  exit 1;\n"
      "fi");

  Future<TaskStatus> statusRunning2, statusFinished2;
  EXPECT_CALL(sched, statusUpdate(_, _))
    .WillOnce(FutureArg<1>(&statusRunning2))
    .WillOnce(FutureArg<1>(&statusFinished2));

  driver.launchTasks(offers2.get()[0].id(), {task2});

  AWAIT_READY(statusRunning2);
  ASSERT_EQ(TASK_RUNNING, statusRunning2->state());

  AWAIT_READY(statusFinished2);
  ASSERT_EQ(TASK_FINISHED, statusFinished2->state());

  driver.stop();
  driver.join();
}


// This test verifies correct failure semantics when
// a task requests a fractional number of GPUs.
TEST_F(NvidiaGpuTest, ROOT_CGROUPS_NVIDIA_GPU_FractionalResources)
{
  Try<Owned<cluster::Master>> master = StartMaster();
  ASSERT_SOME(master);

  // Turn on Nvidia GPU isolation.
  // Assume at least one GPU is available for isolation.
  slave::Flags flags = CreateSlaveFlags();
  flags.isolation = "cgroups/devices,gpu/nvidia";
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:1";

  Owned<MasterDetector> detector = master.get()->createDetector();

  Try<Owned<cluster::Slave>> slave = StartSlave(detector.get(), flags);
  ASSERT_SOME(slave);

  MockScheduler sched;

  FrameworkInfo frameworkInfo = DEFAULT_FRAMEWORK_INFO;
  frameworkInfo.add_capabilities()->set_type(
      FrameworkInfo::Capability::GPU_RESOURCES);

  MesosSchedulerDriver driver(
      &sched, frameworkInfo, master.get()->pid, DEFAULT_CREDENTIAL);

  Future<Nothing> schedRegistered;
  EXPECT_CALL(sched, registered(_, _, _))
    .WillOnce(FutureSatisfy(&schedRegistered));

  Future<vector<Offer>> offers;
  EXPECT_CALL(sched, resourceOffers(_, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return());      // Ignore subsequent offers.

  driver.start();

  AWAIT_READY(schedRegistered);

  // Launch a task requesting a fractional number
  // of GPUs and verify that it fails as expected.
  AWAIT_READY(offers);
  EXPECT_EQ(1u, offers->size());

  TaskInfo task = createTask(
      offers.get()[0].slave_id(),
      Resources::parse("cpus:0.1;mem:128;gpus:0.1").get(),
      "true");

  Future<TaskStatus> status;
  EXPECT_CALL(sched, statusUpdate(_, _))
    .WillOnce(FutureArg<1>(&status));

  driver.launchTasks(offers.get()[0].id(), {task});

  AWAIT_READY(status);

  EXPECT_EQ(TASK_ERROR, status->state());
  EXPECT_EQ(TaskStatus::REASON_TASK_INVALID, status->reason());
  EXPECT_TRUE(strings::contains(
      status->message(),
      "The 'gpus' resource must be an unsigned integer"));

  driver.stop();
  driver.join();
}


TEST_F(NvidiaGpuTest, NVIDIA_GPU_Discovery)
{
  ASSERT_TRUE(nvml::isAvailable());
  ASSERT_SOME(nvml::initialize());

  Try<unsigned int> gpus = nvml::deviceGetCount();
  ASSERT_SOME(gpus);

  slave::Flags flags = CreateSlaveFlags();
  flags.resources = "cpus:1"; // To override the default with gpus:0.
  flags.isolation = "gpu/nvidia";

  Try<Resources> resources = Containerizer::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_SOME(resources->gpus());
  ASSERT_EQ(gpus.get(), resources->gpus().get());
}


// Ensures that the --resources and --nvidia_gpu_devices
// flags are correctly validated.
TEST_F(NvidiaGpuTest, ROOT_CGROUPS_NVIDIA_GPU_FlagValidation)
{
  ASSERT_TRUE(nvml::isAvailable());
  ASSERT_SOME(nvml::initialize());

  Try<unsigned int> gpus = nvml::deviceGetCount();
  ASSERT_SOME(gpus);

  // Not setting the `gpu/nvidia` isolation flag
  // should not trigger-autodiscovery!
  slave::Flags flags = CreateSlaveFlags();

  Try<Resources> resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_NONE(resources->gpus());

  // Setting `--nvidia_gpu_devices` without the `gpu/nvidia`
  // isolation flag should trigger an error.
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:1";

  resources = Containerizer::resources(flags);

  ASSERT_ERROR(resources);

  // Setting GPUs without the `gpu/nvidia` isolation
  // flag should just pass them through without an error.
  flags = CreateSlaveFlags();
  flags.resources = "gpus:100";

  resources = Containerizer::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_SOME(resources->gpus());
  ASSERT_EQ(100u, resources->gpus().get());

  // Setting the `gpu/nvidia` isolation
  // flag should trigger autodiscovery.
  flags = CreateSlaveFlags();
  flags.resources = "cpus:1"; // To override the default with gpus:0.
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_SOME(resources->gpus());
  ASSERT_EQ(gpus.get(), resources->gpus().get());

  // Setting the GPUs to 0 should not trigger auto-discovery!
  flags = CreateSlaveFlags();
  flags.resources = "gpus:0";
  flags.isolation = "gpu/nvidia";

  resources = Containerizer::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_NONE(resources->gpus());

  // --nvidia_gpu_devices and --resources agree on the number of GPUs.
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:1";
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_SOME(resources);
  ASSERT_SOME(resources->gpus());
  ASSERT_EQ(1u, resources->gpus().get());

  // Both --resources and --nvidia_gpu_devices must be specified!
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "cpus:1"; // To override the default with gpus:0.
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);

  flags = CreateSlaveFlags();
  flags.resources = "gpus:" + stringify(gpus.get());
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);

  // --nvidia_gpu_devices and --resources do not match!
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:2";
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);

  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});
  flags.resources = "gpus:0";
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);

  // More than available on the machine!
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>();
  flags.resources = "gpus:" + stringify(2 * gpus.get());
  flags.isolation = "gpu/nvidia";

  for (size_t i = 0; i < 2 * gpus.get(); ++i) {
    flags.nvidia_gpu_devices->push_back(i);
  }

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);

  // Set `nvidia_gpu_devices` to contain duplicates.
  flags = CreateSlaveFlags();
  flags.nvidia_gpu_devices = vector<unsigned int>({0u, 0u});
  flags.resources = "cpus:1;gpus:1";
  flags.isolation = "gpu/nvidia";

  resources = NvidiaGpuAllocator::resources(flags);

  ASSERT_ERROR(resources);
}


// Test proper allocation / deallaoction of GPU devices.
TEST_F(NvidiaGpuAllocatorTest, NVIDIA_GPU_VerifyAllocation)
{
  ASSERT_TRUE(nvml::isAvailable());
  ASSERT_SOME(nvml::initialize());

  slave::Flags flags = CreateSlaveFlags();
  flags.resources = "cpus:1"; // To override the default with gpus:0.
  flags.isolation = "gpu/nvidia";

  Try<NvidiaGpuAllocator*> _allocator = NvidiaGpuAllocator::create(flags);
  ASSERT_SOME(_allocator);

  Owned<NvidiaGpuAllocator> allocator(_allocator.get());

  Try<unsigned int> total = nvml::deviceGetCount();
  ASSERT_SOME(total);
  ASSERT_GE(total.get(), 1u);

  // Allocate all GPUs at once.
  Future<Option<set<Gpu>>> gpus = allocator->allocate(total.get());

  AWAIT_ASSERT_READY(gpus);
  ASSERT_SOME(gpus.get());
  ASSERT_EQ(total.get(), gpus->get().size());

  // Make sure there are no GPUs left to allocate.
  Future<Option<set<Gpu>>> gpu = allocator->allocate(1);

  AWAIT_ASSERT_READY(gpu);
  ASSERT_NONE(gpu.get());

  // Free all GPUs at once and reallocate them by reference.
  Future<Nothing> result = allocator->deallocate(gpus->get());
  AWAIT_ASSERT_READY(result);

  result = allocator->allocate(gpus->get());

  AWAIT_ASSERT_READY(result);

  // Free 1 GPU back and reallocate it. Make sure they are the same.
  result = allocator->deallocate({*gpus->get().begin()});

  AWAIT_ASSERT_READY(result);

  gpu = allocator->allocate(1);

  AWAIT_ASSERT_READY(gpu);
  ASSERT_SOME(gpu.get());
  ASSERT_EQ(*gpus->get().begin(), *gpu->get().begin());

  // Attempt to free the same GPU twice.
  result = allocator->deallocate({*gpus->get().begin()});
  AWAIT_ASSERT_READY(result);

  result = allocator->deallocate({*gpus->get().begin()});
  AWAIT_ASSERT_FAILED(result);

  // Allocate a specific GPU by reference.
  result = allocator->allocate({*gpus->get().begin()});
  AWAIT_ASSERT_READY(result);

  // Attempt to free a bogus GPU.
  result = allocator->deallocate({Gpu()});
  AWAIT_ASSERT_FAILED(result);

  // Free all GPUs.
  result = allocator->deallocate(gpus->get());
  AWAIT_ASSERT_READY(result);

  // Attempt to allocate a bogus GPU.
  result = allocator->allocate({Gpu()});
  AWAIT_ASSERT_FAILED(result);
}


// Nvidia GPU provision test with docker containerizer.
TEST_F(NvidiaGpuDockerContainerizerTest, ROOT_DOCKER_LaunchWithGpu)
{
  ASSERT_TRUE(nvml::isAvailable());
  ASSERT_SOME(nvml::initialize());

  Try<Owned<cluster::Master>> master = StartMaster();
  ASSERT_SOME(master);

  MockDocker* mockDocker =
    new MockDocker(tests::flags.docker, tests::flags.docker_socket);

  Shared<Docker> docker(mockDocker);

  slave::Flags flags = CreateSlaveFlags();
  flags.resources = "gpus:1";
  flags.isolation = "cgroups/devices,gpu/nvidia";
  flags.nvidia_gpu_devices = vector<unsigned int>({0u});

  Try<NvidiaGpuAllocator*> _allocator = NvidiaGpuAllocator::create(flags);
  ASSERT_SOME(_allocator);

  Shared<NvidiaGpuAllocator> allocator(_allocator.get());

  // Make sure GPU number > 1 for following tests.
  ASSERT_NE(0u, allocator.get()->allGpus().size());

  Fetcher fetcher;

  Try<ContainerLogger*> logger =
    ContainerLogger::create(flags.container_logger);

  ASSERT_SOME(logger);

  MockDockerContainerizer dockerContainerizer(
      flags,
      &fetcher,
      Owned<ContainerLogger>(logger.get()),
      docker,
      allocator);

  Owned<MasterDetector> detector = master.get()->createDetector();

  Try<Owned<cluster::Slave>> slave =
    StartSlave(detector.get(), &dockerContainerizer, flags);
  ASSERT_SOME(slave);

  MockScheduler sched;

  FrameworkInfo frameworkInfo = DEFAULT_FRAMEWORK_INFO;
  frameworkInfo.add_capabilities()->set_type(
      FrameworkInfo::Capability::GPU_RESOURCES);

  MesosSchedulerDriver driver(
      &sched, frameworkInfo, master.get()->pid, DEFAULT_CREDENTIAL);

  Future<FrameworkID> frameworkId;
  EXPECT_CALL(sched, registered(&driver, _, _))
    .WillOnce(FutureArg<1>(&frameworkId));

  Future<vector<Offer> > offers;
  EXPECT_CALL(sched, resourceOffers(&driver, _))
    .WillOnce(FutureArg<1>(&offers))
    .WillRepeatedly(Return()); // Ignore subsequent offers.

  driver.start();

  AWAIT_READY(frameworkId);

  AWAIT_READY(offers);
  ASSERT_NE(0u, offers.get().size());

  const Offer& offer = offers.get()[0];

  SlaveID slaveId = offer.slave_id();

  TaskInfo task;
  task.set_name("");
  task.mutable_task_id()->set_value("1");
  task.mutable_slave_id()->CopyFrom(offer.slave_id());
  task.mutable_resources()->CopyFrom(offer.resources());

  CommandInfo command;
  command.set_value("sleep 1000");

  ContainerInfo containerInfo;
  containerInfo.set_type(ContainerInfo::DOCKER);

  // TODO(tnachen): Use local image to test if possible.
  ContainerInfo::DockerInfo dockerInfo;
  dockerInfo.set_image("alpine");
  containerInfo.mutable_docker()->CopyFrom(dockerInfo);

  task.mutable_command()->CopyFrom(command);
  task.mutable_container()->CopyFrom(containerInfo);

  Future<ContainerID> containerId;
  EXPECT_CALL(dockerContainerizer, launch(_, _, _, _, _, _, _, _))
    .WillOnce(DoAll(FutureArg<0>(&containerId),
                    Invoke(&dockerContainerizer,
                           &MockDockerContainerizer::_launch)));

  Future<TaskStatus> statusRunning;
  EXPECT_CALL(sched, statusUpdate(&driver, _))
    .WillOnce(FutureArg<1>(&statusRunning))
    .WillRepeatedly(DoDefault());

  driver.launchTasks(offers.get()[0].id(), {task});

  AWAIT_READY_FOR(containerId, Seconds(60));
  AWAIT_READY_FOR(statusRunning, Seconds(60));
  EXPECT_EQ(TASK_RUNNING, statusRunning.get().state());
  ASSERT_TRUE(statusRunning.get().has_data());

  Try<JSON::Array> array = JSON::parse<JSON::Array>(statusRunning.get().data());
  ASSERT_SOME(array);

  // Check if container information is exposed through master's state endpoint.
  Future<http::Response> response = http::get(
      master.get()->pid,
      "state",
      None(),
      createBasicAuthHeaders(DEFAULT_CREDENTIAL));

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(process::http::OK().status, response);

  Try<JSON::Object> parse = JSON::parse<JSON::Object>(response.get().body);
  ASSERT_SOME(parse);

  Result<JSON::Value> find = parse.get().find<JSON::Value>(
      "frameworks[0].tasks[0].container.docker.privileged");

  EXPECT_SOME_FALSE(find);

  // Check if container information is exposed through slave's state endpoint.
  response = http::get(
      slave.get()->pid,
      "state",
      None(),
      createBasicAuthHeaders(DEFAULT_CREDENTIAL));

  AWAIT_EXPECT_RESPONSE_STATUS_EQ(process::http::OK().status, response);

  parse = JSON::parse<JSON::Object>(response.get().body);
  ASSERT_SOME(parse);

  find = parse.get().find<JSON::Value>(
      "frameworks[0].executors[0].tasks[0].container.docker.privileged");

  EXPECT_SOME_FALSE(find);

  // Now verify that the TaskStatus contains the container IP address.
  ASSERT_TRUE(statusRunning.get().has_container_status());
  EXPECT_EQ(1, statusRunning.get().container_status().network_infos().size());
  EXPECT_EQ(1, statusRunning.get().container_status().network_infos(0).ip_addresses().size()); // NOLINT(whitespace/line_length)

  ASSERT_TRUE(exists(docker, slaveId, containerId.get()));

  // Now verify that GPU0 is exposed to container.
  string name = containerName(slaveId, containerId.get());
  Future<Docker::Container> inspect = docker->inspect(name);

  AWAIT_READY(inspect);

  string results = inspect.get().output;

  // Get allocated GPU from allocator. We assume that the allocator
  // allocated the first GPU to the container.
  unsigned int minor = allocator.get()->allGpus().begin()->minor;

  Option<set<string>> deviceInspect = parseInspectDevices(results);

  // Check if nvidia GPU devices are successfully exposed.
  ASSERT_SOME(deviceInspect);
  ASSERT_NE(0, deviceInspect.get().count("/dev/nvidiactl"));
  ASSERT_NE(0, deviceInspect.get().count("/dev/nvidia-uvm"));
  ASSERT_NE(0, deviceInspect.get().count("/dev/nvidia" + stringify(minor)));

  Future<containerizer::Termination> termination =
    dockerContainerizer.wait(containerId.get());

  driver.stop();
  driver.join();

  AWAIT_READY(termination);

  ASSERT_FALSE(
    exists(docker, slaveId, containerId.get(), ContainerState::RUNNING));
}

} // namespace tests {
} // namespace internal {
} // namespace mesos {
