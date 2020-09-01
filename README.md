# MegRay

MegRay is a cross-platform communication library providing point-to-point and collective communication methods, such as send, recv, all\_gather, all\_reduce, reduce\_scatter, reduce, broadcast, gather, scatter and all\_to\_all. In the area of deep learning, these methods can be utilized for implementing distributed training framework, including data parallel and model parallel. Currently there are three backends, nccl and ucx for cuda platform, rccl for rocm platform. In the future, algorithms on more platforms will be added.

## Build

0. prepare third party repositories.

```
./third_party/prepare.sh
```

1. Make a directory for build.

```
mkdir build
cd build
```

2. Generate build configurations by `CMake`.

```
cmake .. -DMEGRAY_WITH_TEST=ON
```

3. Start to build
```
make
```
