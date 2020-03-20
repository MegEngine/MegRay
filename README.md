# MegRay

MegRay is a cross-platform communication library providing point-to-point and collective communication methods, such as send, recv, all\_gather, all\_reduce, reduce\_scatter, reduce and broadcast. In the area of deep learning, these methods can be utilized for implementing distributed training framework, including data parallel and model parallel. Currently there are two backends, nccl and ucx, and only cuda platform is supported. In the future, algorithms on more platforms will be added.

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
cmake .. -DMEGRAY_TEST=ON
```

3. Start to build
```
make
```
