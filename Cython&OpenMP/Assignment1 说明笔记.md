# Assignment1 说明笔记

**学号：G20200389020037**



在Assignment1 中，我分别使用了Cython和OpenMP完成了target_encoding代码块的加速工作。



## Cython

### 算法描述

在最终的Cython 加速版本中，我尝试将原先的二维数组输入data 拆分成两个独立的数组 data_x 和data_y，这样能避免在函数传递指针的过程中出现数组内存不连续的问题。介于data_x 和data_y 元素的取值范围，我将传入的两个一维数组的数据类型指定为int，而不是默认的long。

对于局部变量value_dict 和count_dict，我选择使用C++类型数据map / unordered_dict 代替。

在两个for 循环中，我使用了并行化方法prange，并将nogil置为True，即在for循环中使用并行化计算。



### 加速结果

在target_encoding 的v1 版本中，执行target_mean_v1 函数，数据长度为5000 时，运行时间为25.6 s，可以看出，pandas 处理数据的效率较低；使用numpy 处理数据，即在target_encoding 的v2 版本中，执行target_mean_v2 函数，相同的输入数据下，运行时间为 264 ms，速度约为v1 的97 倍。使用Cython 加速后，最终target_mean_v8 函数的运行时间为 413 us，速度约为v1 的62000 倍。可以看出，Cython 加速效果相较于numpy，有进一步的显著提升。



## OpenMP

### 算法描述

在最终的OpenMP加速版本中，使用C++ 语言编译来实现OpenMP 并行加速。在scratch.pyx 文件中，主要函数target_mean_v8_parallel 使用了nogil 进行加速。



### 加速结果

通过执行setup.py install 后生成Python Extension Module，执行测试函数可以发现，在不考虑输出时间和输入数据初始化时间的前提下，函数的运行时间约为306 us，速度约为v1 的83700 倍。可以看出，OpenMP 并行计算加速效果优于纯Cython 加速。