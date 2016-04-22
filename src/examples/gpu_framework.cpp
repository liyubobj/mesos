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

#include <iostream>
#include <string>

#include <mesos/resources.hpp>
#include <mesos/scheduler.hpp>

#include <process/clock.hpp>
#include <process/dispatch.hpp>
#include <process/process.hpp>

#include <stout/flags.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/path.hpp>
#include <stout/stringify.hpp>

#include "logging/flags.hpp"
#include "logging/logging.hpp"

using namespace mesos;

using process::Clock;
using process::Process;
using process::Timer;

using std::string;
using std::vector;

using mesos::Resources;

const int32_t CPUS_PER_TASK = 1;
const int32_t MEM_PER_TASK = 128;


class GpuScheduler : public Scheduler
{
public:
  GpuScheduler(const int32_t _numGpus, const Duration& _timeout)
    : numGpus(_numGpus),
      timeout(_timeout),
      offerAccepted(false) {}

  virtual ~GpuScheduler() {}

  virtual void registered(SchedulerDriver* driver,
                          const FrameworkID& frameworkId,
                          const MasterInfo& masterInfo)
  {
    LOG(INFO) << "Registered: " << frameworkId;

    timer = clock.timer(timeout, [=]() {
      LOG(ERROR) << "Timeout waiting for offer with GPU Resources:"
                 << " Waited " << timeout;
      driver->abort();
    });
  }

  virtual void reregistered(SchedulerDriver*, const MasterInfo& masterInfo)
  {
    LOG(INFO) << "Reregistered";
  }

  virtual void disconnected(SchedulerDriver* driver)
  {
    LOG(INFO) << "Disconnected";
  }

  virtual void resourceOffers(SchedulerDriver* driver,
                              const vector<Offer>& offers)
  {
    foreach (const Offer& offer, offers) {
      // We only accept one offer
      if (!offerAccepted && clock.cancel(timer)) {
        LOG(INFO) << "Received offer " << offer.id()
                  << " with " << offer.resources();

        static const Resources TASK_RESOURCES = Resources::parse(
            "gpus:" + stringify(numGpus) +
            ";cpus:" + stringify(CPUS_PER_TASK) +
            ";mem:" + stringify(MEM_PER_TASK)).get();

        LOG(INFO) << "Starting the task";

        CommandInfo commandInfo;
        commandInfo.set_shell(true);
        commandInfo.set_value("nvidia-smi");

        TaskInfo task;
        task.set_name("GPU Task");
        task.mutable_task_id()->set_value("0");
        task.mutable_slave_id()->MergeFrom(offer.slave_id());
        task.mutable_command()->MergeFrom(commandInfo);

        Option<Resources> resources =
          Resources(offer.resources()).find(TASK_RESOURCES.flatten("*"));

        CHECK_SOME(resources);
        task.mutable_resources()->MergeFrom(resources.get());

        driver->launchTasks(offer.id(), {task});

        offerAccepted = true;
      }
    }
  }

  virtual void offerRescinded(SchedulerDriver* driver, const OfferID& offerId)
  {
    LOG(INFO) << "Offer Rescinded: " << offerId;
  }

  virtual void statusUpdate(SchedulerDriver* driver, const TaskStatus& status)
  {
    string taskId = status.task_id().value();

    LOG(INFO) << "Task " << taskId << " is in state " << status.state();

    if (status.state() == TASK_LOST ||
        status.state() == TASK_KILLED ||
        status.state() == TASK_FAILED) {
      LOG(ERROR) << "Aborting because task " << taskId
                 << " is in unexpected state " << status.state()
                 << " with reason " << status.reason()
                 << " from source " << status.source()
                 << " with message '" << status.message() << "'";
      driver->abort();
    } else if (status.state() == TASK_FINISHED) {
      driver->stop();
    }
  }

  virtual void frameworkMessage(SchedulerDriver* driver,
                                const ExecutorID& executorId,
                                const SlaveID& slaveId,
                                const string& data)
  {
    LOG(INFO) << "Framework Message (ExecutorId " << executorId
              << ", SlaveId " << slaveId << "): " << data;
  }

  virtual void slaveLost(SchedulerDriver* driver, const SlaveID& sid)
  {
    LOG(INFO) << "Slave Lost: " << sid;
  }

  virtual void executorLost(SchedulerDriver* driver,
                            const ExecutorID& executorId,
                            const SlaveID& slaveId,
                            int status)
  {
    LOG(INFO) << "Executor Lost (ExecutorId " << executorId
              << ", SlaveId " << slaveId << "): " << status;
  }

  virtual void error(SchedulerDriver* driver, const string& message)
  {
    LOG(ERROR) << message;
  }

private:
  int32_t numGpus;
  Duration timeout;
  bool offerAccepted;
  Clock clock;
  Timer timer;
};


void usage(const char* argv0, const flags::FlagsBase& flags)
{
  std::cerr << "Usage: " << Path(argv0).basename() << " [...]" << std::endl;
  std::cerr << "Supported options:" << std::endl;
  std::cerr << flags.usage() << std::endl;
}


int main(int argc, char** argv)
{
  mesos::internal::logging::Flags flags;

  Option<string> master;
  flags.add(&master,
            "master",
            "ip:port of master to connect");

  bool allowGpus;
  flags.add(&allowGpus,
            "allow_gpus",
            "Allow this framework to receive GPU resources",
            true);

  int32_t numGpus;
  flags.add(&numGpus,
            "num_gpus",
            "Number of GPUs to request",
            0);

  int32_t timeout;
  flags.add(&timeout,
            "timeout",
            "Number of seconds to wait for an offer with GPUs before exiting",
            10);

  Try<flags::Warnings> load = flags.load(None(), argc, argv);

  if (load.isError()) {
    LOG(ERROR) << load.error();
    usage(argv[0], flags);
    exit(EXIT_FAILURE);
  } else if (master.isNone()) {
    LOG(ERROR) << "Missing --master";
    usage(argv[0], flags);
    exit(EXIT_FAILURE);
  }

  internal::logging::initialize(argv[0], flags, true); // Catch signals.

  FrameworkInfo framework;
  framework.set_user(""); // Have Mesos fill in the current user.
  framework.set_name("Test GPUs (C++)");
  framework.set_principal("gpu-framework-cpp");

  if (allowGpus) {
    framework.add_capabilities()->set_type(
        FrameworkInfo::Capability::GPU_RESOURCES);
  }

  MesosSchedulerDriver* driver;
  GpuScheduler scheduler(numGpus, Seconds(timeout));

  driver = new MesosSchedulerDriver(
      &scheduler,
      framework,
      master.get());

  int status = driver->run() == DRIVER_STOPPED ? 0 : 1;

  // Ensure that the driver process terminates.
  driver->stop();
  delete driver;
  return status;
}
