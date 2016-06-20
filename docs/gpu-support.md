---
title: Apache Mesos - Nvidia GPU Support
layout: documentation
---

# Nvidia GPU Support

Mesos 1.0.0 added experimental support for Nvidia GPUs.

## Getting Started

There are three primary groups of people to consider when documenting
how to work with Nvidia GPUs in Mesos:

* [Mesos Developers](#mesos-developers)
* [Mesos Users](#mesos-users)
* [Mesos Cluster Operators](#mesos-cluster-operators)

**Mesos developers** are people who build the Mesos codebase directly
from source. These people need to make sure their build environment is
set up properly to enable building Mesos binaries with proper Nvidia
GPU support.

**Mesos users** are framework developers or people launching tasks
that consume resources in a Mesos cluster. These people need to know
how to consume GPUs so that the jobs they schedule actually run on
machines equipped with Nvidia GPUs.

**Mesos cluster operators** are people who operate a running Mesos
cluster. These people need to know what flags to pass to a Mesos agent
to take advantage of Nvidia GPU support in Mesos. They also need to
know what external software must be installed on the host machines
where these agents are launched in order to take advantage of this
support.

In the sections below, we outline the important information that each
group needs to know about. You can click on the group name above to
jump immediately to the relevant documentation for that group.

**Note:** Nvidia GPU support in Mesos is still experimental and has a
number of limitations you should be aware of before getting started.
Please see the section below on ['Limitations'](#limitations) for more
details.

<a name="mesos-developers"></a>
## Mesos Developers

As a Mesos developer, you will need to install some external software
as well as configure Mesos appropriately in order to build Nvidia GPU
support into Mesos. Additionally, a new `NVIDIA_GPU` unit test filter
has been introduced in order to run Nvidia GPU unit tests on Nvidia
GPU equipped machines.

The sections below outline everything necessary to get you started.

### The Nvidia Management Library (NVML)
Building Mesos with Nvidia GPU support has an external dependency not
covered by the standard [Mesos Getting Started
Guide](getting-started.md). Specifically, it depends on the API
defined by the [Nvidia Management Library
(NVML)](https://developer.nvidia.com/nvidia-management-library-nvml),
which Mesos uses to manage Nvidia GPU devices installed on its
agents.

For development purposes, NVML consists of a single header file
(`nvml.h`) and a *stub* library (`libnvidia-ml.so`), included as part
of the [Nvidia GPU Deployment Kit
(GDK)](https://developer.nvidia.com/gpu-deployment-kit).

The idea behind the GDK is to allow developers to write code against
its stub library on machines **not** equipped with Nvidia GPUs and
then later move those binaries to production machines and link them
with a real version of `libnvidia-ml.so`.
Ideally, the GDK would not be unnecessary if developing on a machine
that already included the real version of `libnvidia-ml.so`. However,
the only way to currently obtain the `nvml.h` header file is to
install the full-blown GDK, even though the stub library it includes
is unnecessary.

We provide a helper script to install the Nvidia GDK as:

    [mesos]$ ./support/install-nvidia-gdk.sh

**Note:** This script only works on Linux.

This script takes two optional parameters:

    --install-dir=<path>

    --update-ldcache

By default, this script will attempt to install the Nvidia GDK into
`/opt/nvidia-gdk`. You can optionally override this with the
`--install-dir` option. If you choose to install the GDK elsewhere, we
*highly* recommend installing it into a self-contained directory. This
makes uninstalling the GDK in the future much easier (all you need to
do is delete the top-level directory instead of trying to find
individual files spread across the file system and delete them
manually).

Specifying the `--update-ldcache` flag will install a file called
`nvidia-gdk.conf` into your `/etc/ld.so.conf.d/` directory, followed
by a call to `sudo ldconfig`. This will cause your ld-cache to be
updated with the path of the `libnvidia-ml.so` library so you don't
have to continuously set `LD_LIBARY_PATH` every time you want to run a
mesos binary. If you choose to run the script with this flag, make
sure to remove this file and rerun `ldconfig` when uninstalling the
GDK in the future.

### Configuring Mesos

Once the Nvidia GDK is installed, Mesos can be configured to build
with Nvidia GPU support. The following configure flags must be set to
enable this:

    --enable-nvidia-gpu-support

    --with-nvml-include=<path_to_include>

    --with-nvml-lib=<path_to_lib>

The `--enable-nvidia-gpu-support` flag enables GPU support in Mesos,
and **must** be set in order for Mesos to treat GPUs as resources.

The `--with-nvml-include` flag specifies the path to the top-level
include directory inside the Nvidia GDK installation. If you installed
the GDK to its default location, this path will be:

    /opt/nvidia-gdk/usr/include

The `--with-nvml-lib` flag specifies the path to the lib directory
containing `libnvidia-ml.so` inside the Nvidia GDK installation. If
you installed the GDK to its default location, this path will be:

    /opt/nvidia-gdk/usr/src/gdk/nvml/lib

With these three flags combined, a typical call to configure Mesos
with Nvidia GPU support might look like:

    [mesos/build]$
     ../configure --enable-nvidia-gpu-support \
                  --with-nvml-include=/opt/nvidia-gdk/usr/include \
                  --with-nvml-lib=/opt/nvidia-gdk/usr/src/gdk/nvml/lib

From here, you should be able to build and install mesos normally:

    [mesos/build]$ make -j
    [mesos/build]$ make -j install

<a name="unit-tests"></a>
### Running Unit Tests

At the time of this writing, the following Nvidia GPU specific unit
tests exist:

    NvidiaGpuTest.ROOT_CGROUPS_NVIDIA_GPU_VerifyDeviceAccess
    NvidiaGpuTest.ROOT_CGROUPS_NVIDIA_GPU_FractionalResources

The capitalized words following the `'.'` specify test filters to
apply when running the unit tests. In our case the filters that apply
are `ROOT`, `CGROUPS`, and `NVIDIA_GPU`. This means that these tests
must be run as `root` on Linux machines with `cgroups` support that
have Nvidia GPUs installed on them. The check to verify that Nvidia
GPUs exist is to look for the existence of the Nvidia System
Management Interface (`nvidia-smi`) on the machine where the tests are
being run. This binary must be installed on any Nvidia GPU equipped
machine in order to run Mesos on it. Please refer to the section on
[Mesos Cluster Operators](#mesos-cluster-operators) for more
information.

So long as these filters are satisfied, you can run the following to
execute these unit tests:

    [mesos]$ GTEST_FILTER="" make -j check
    [mesos]$ sudo bin/mesos-tests.sh --gtest_filter="*NVIDIA_GPU*"

<a name="mesos-users"></a>
## Mesos Users

As a Mesos user, you want to launch tasks that consume Nvidia GPUs.
To do so, Mesos exposes GPUs as a simple `SCALAR` resource in the same
way it always has for CPUs, memory, and disk. That is, a resource
offer such as the following is now possible:

    cpus:8; mem:1024; disk:65536; gpus:4;

When a scheduler receives such an offer it is free to allocate any
subset of the offered GPUs to its tasks in the same way it does for
CPUs, memory, and disk. However, unlike CPUs, memory, and disk, *only*
whole numbers of GPUs can be selected. If a fractional amount is
selected, launching the task will result in a `TASK_ERROR`.

For example, a simple C++ scheduler that uses the default
`CommandExecutor` might implement its `resourceOffers()` callback to
launch a task that consumes GPUs as follows:

    virtual void resourceOffers(SchedulerDriver* driver,
                                const std::vector<Offer>& offers)
    {
      std::cout << "Resource offers received" << std::endl;

      for (size_t i = 0; i < offers.size(); i++) {
        const Offer& offer = offers[i];

        // We just launch one task.
        if (!taskLaunched) {
          double cpus = getScalarResource(offer, "cpus");
          double mem = getScalarResource(offer, "mem");
          double gpus = getScalarResource(offer, "gpus");

          if (cpus < 0.1 || mem < 128 || gpus < 1) {
            continue;
          }

          std::cout << "Starting the task" << std::endl;

          TaskInfo task1 = createTask(
              offer.slave_id(),
              Resources::parse("cpus:0.1;mem:128;gpus:1;").get(),
              "nvidia-smi");

          driver->launchTasks(offer.id(), {task});

          taskLaunched = true;
        }
      }
    }

This example accepts a single offer to run the `nvidia-smi` command
with access to to 0.1 CPUs, 128 MB of RAM, and 1 GPU. More
sophisticated logic can be added to do more interesting things, but
the main point here is that GPUs are treated the same as any other
`SCALAR` resource. Under the hood, Mesos will ensure that only those
GPUs allocated to a task will be accessible by it.

Please refer to the section on [Limitations](#limitations) for a list
of limitations that currently exist with supporting Nvidia GPUs in
this way.

<a name="mesos-cluster-operators"></a>
## Mesos Cluster Operators

As a Mesos cluster operator, you need to ensure that the proper set of
flags have been set for Nvidia GPU support when launching Mesos
agents. You also need to ensure that the proper external dependencies
are installed on the machines those agents are running on. Currently
we only support running these agents on Linux.

<a name="agent-flags"></a>
### Agent Flags
There are three agent flags of importance to enable Nvidia GPU support
in Mesos. The only one technicaly necessary is the `--isolation` flag,
but the others are useful to restrict access to a subset of GPUs on
the machine.

    --isolation="cgroups/devices,gpus/nvidia"

    --nvidia_gpu_devices="<list_of_gpu_ids>"

    --resources="gpus:<num_gpus>"

For the `--isolation` flag, you need to ensure that you enable *both*
the `cgroups/devices` isolator *and* the `gpus/nvidia` isolator in
order to properly isolate access to GPUs. Under the hood, Mesos will
use the Linux `cgroups` devices subsystem to enforce the necessary GPU
isolation between tasks.

For the `--nvidia-gpu-devices` flag, you need to provide a comma
separated list of GPUs, as determined by running `nvidia-smi` on the
host where the agent is to be launched ([see
below](#external-dependencies) for instructions on what external
dependencies must be installed on these hosts). A common output from
`nvidia-smi` listing four GPUs can be seen below:

    +------------------------------------------------------+
    | NVIDIA-SMI 352.79     Driver Version: 352.79         |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla M60           Off  | 0000:04:00.0     Off |                    0 |
    | N/A   34C    P0    39W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla M60           Off  | 0000:05:00.0     Off |                    0 |
    | N/A   35C    P0    39W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla M60           Off  | 0000:83:00.0     Off |                    0 |
    | N/A   38C    P0    40W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla M60           Off  | 0000:84:00.0     Off |                    0 |
    | N/A   34C    P0    39W / 150W |     34MiB /  7679MiB |     97%      Default |
    +-------------------------------+----------------------+----------------------+

The GPU `id` to choose can be seen in the far left of each row. Any
subset of these `ids` can be listed in the `--nvidia_gpu_devices`
flag. That is, all of the following values of this flag are valid:

    --nvidia_gpu_devices="0"
    --nvidia_gpu_devices="0,1"
    --nvidia_gpu_devices="0,1,2"
    --nvidia_gpu_devices="0,1,2,3"
    --nvidia_gpu_devices="0,2,3"
    --nvidia_gpu_devices="3,1"
    etc...

For the `--resources=gpus:<num_gpus>` flag, the value passed to
`<num_gpus>` must equal the number of GPUs listed in
`--nvidia-gpu-devices`. If these numbers do not match, launching the
agent will fail.

As mentioned at the beginning of this section, the `--resources` flag
and the `--nvidia-gpu-devices` flag are optional. However, you must
either set *both* of these flags together or set *neither* of them in
order for the agent to start up properly. If neither flag is set, then
Mesos will autodiscover the number of GPUs available on the machine
and advertise them as part of its resource offer. If both flags are
set, then Mesos will use the information provided in the flags to
build its resource offer from the subset of GPUs listed.

<a name="external-dependencies"></a>
### External Dependencies

Any host running a Mesos agent with Nvidia GPU support **MUST** have a
valid Nvidia kernel driver installed. It is also *highly* recommended to
install the corresponding user-level libraries and tools available as
part of the Nvidia CUDA toolkit. Many jobs that use Nvidia GPUs rely
on CUDA and not including it will severely limit the type of
GPU-aware jobs you can run on Mesos.

#### Installing the Required Tools

The Nvidia kernel driver can be downloaded at the link below. Make
sure to choose the proper model of GPU, operating system, and CUDA
toolkit you plan to install on your host:

    http://www.nvidia.com/Download/index.aspx

Unfortunately, most Linux distributions come preinstalled with an open
source video driver called `Nouveau`. This driver conflicts with the
Nvidia driver we are trying to install. The following guides may prove
useful to help guide you through the process of uninstalling `Nouveau`
before installing the Nvidia driver on `CentOS` or `Ubuntu`.

    http://www.dedoimedo.com/computers/centos-7-nvidia.html
    http://www.allaboutlinux.eu/remove-nouveau-and-install-nvidia-driver-in-ubuntu-15-04/

After installing the Nvidia kernel driver, you can follow the
instructions in the link below to install the Nvidia CUDA toolkit:

    http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/

In addition to the steps listed in the link above, it is *highly*
recommended to add CUDA's `lib` directory into your `ldcache` so that
tasks launched by Mesos will know where these libraries exist and link
with them properly.

    sudo bash -c "cat > /etc/ld.so.conf.d/cuda-lib64.conf << EOF
    /usr/local/cuda/lib64
    EOF"

    sudo ldconfig

If you choose **not** to add CUDAs `lib` directory to your `ldcache`,
you **MUST** set the `LD_LIBRARY_PATH` to the `lib` directory
every time you launch an agent.

    LD_LIBRARY_PATH=/usr/local/cuda/lib64 mesos-agent --master=<master_ip>:5050

**Note:** This is *not* the recommended method. You have been warned.

#### Verifying the Installation

Once the kernel driver has been installed, you can make sure
everything is working by trying to run the bundled `nvidia-smi` tool.

    nvidia-smi

You should see output similar to the following:

    Thu Apr 14 11:58:17 2016
    +------------------------------------------------------+
    | NVIDIA-SMI 352.79     Driver Version: 352.79         |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla M60           Off  | 0000:04:00.0     Off |                    0 |
    | N/A   34C    P0    39W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla M60           Off  | 0000:05:00.0     Off |                    0 |
    | N/A   35C    P0    39W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla M60           Off  | 0000:83:00.0     Off |                    0 |
    | N/A   38C    P0    38W / 150W |     34MiB /  7679MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla M60           Off  | 0000:84:00.0     Off |                    0 |
    | N/A   34C    P0    38W / 150W |     34MiB /  7679MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

To verify your CUDA installation, it is recommended to go through the instructions at the link below:

    http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#install-samples

Finally, you should get a developer to run the Nvidia GPU related unit
tests on your machine to ensure that everything passes. The details of
running these unit tests can be found [here](#unit-tests).

<a name="limitations"></a>
## Limitations

The current GPU support in Mesos has a number of limitations:

* Nvdida GPU support is only available on Linux. Mesos uses the Linux
  `cgroups` devices subsystem to enforce the necessary GPU isolation
   between tasks.

* Filesystem isolation must be **disabled**. This means **No Docker
  Support** at the moment. Only the mesos containerizer can currently
  be used.

* There is no GPU autodiscovery. The Mesos Cluster Operator must
  manually list out the available GPUs via agent flags at the time
  they launch an agent (See [Agent Flags](#agent-flags) for more
  information).

* As a user, there is no way to determine any information about what
  GPUs are available to you. Resource offers simply contain a *number*
  of GPUs that happen to be available. There is no way to pick an
  exact GPU or discover anything about the topology of available GPUs
  to determine if any of them are collocated.

* There is no way to allocate fractional portions of a GPU to
  different tasks. Newer Nvidia hardware supports some level of
  sharing, but this is currently not exposed by Mesos in any way.

* There is currently no usage information available in the `metrics`
  endpoint about GPU usage. Eventually, we plan to use NVML to gather
  this information and add it to the `ResourceStatistics` protobuf. We
  are still deciding what the appropriate metrics to add are.

We plan to slowly remove these limitations as we add more support to
address them. As limitations are removed, this documentation will be
updated accordingly.
