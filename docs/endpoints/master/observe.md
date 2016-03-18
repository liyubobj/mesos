---
title: Apache Mesos - HTTP Endpoints - /observe
layout: documentation
---
<!--- This is an automatically generated file. DO NOT EDIT! --->

### USAGE ###
>        /observe
>        /master/observe

### TL;DR; ###
Observe a monitor health state for host(s).

### DESCRIPTION ###
This endpoint receives information indicating host(s)
health.
The following fields should be supplied in a POST:
1. monitor - name of the monitor that is being reported
2. hosts - comma separated list of hosts
3. level - OK for healthy, anything else for unhealthy


### AUTHENTICATION ###
This endpoint requires authentication iff HTTP authentication is
enabled.