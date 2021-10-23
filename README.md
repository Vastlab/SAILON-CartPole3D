# SAILON-CartPole3D
Repo for the VAST Lab code for the SAIL-ON Project applied to 3D CartPole++ problem. 

# Usage Instructions

```git clone https://github.com/Vastlab/SAILON-CartPole3D.git```

Once the repo is cloned it is ready to be ran using docker-compose.

```
docker-compose -f uccs-ta2.yml build
docker-compose -f uccs-ta2.yml up -d --scale uccs-ta2=1
docker-compose -f uccs-ta2.yml logs -f --tail=5
```

Once the testing episodes begin you can run more docker containers to speed up testing in parallel.

```
docker-compose -f uccs-ta2.yml up -d --scale uccs-ta2=5
```

When the testing is completed stop the docker containers.

```
docker-compose -f uccs-ta2.yml down
```
