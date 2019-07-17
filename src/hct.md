---
marp: true
---

# 硬件综合训练答辩（FPGA方向）
**组长：薛昕磊**
**组员：李英平 莫冯然 武根泽**
**答辩人：李英平**

---

# 课设过程

---

# 开课之前

---

1. 安装破解Quartus 17.1，搭建完整的开发环境。
2. 熟悉康芯FPGA开发平台和Altera DE1-SoC FPGA芯片。
3. 运行开发平台提供的Demo（交通灯、直流电机、摄像头、触摸屏、超声波、51软核），熟悉FPGA的开发流程。
4. 完成了SOPC实验，熟悉了使用NIOS II软核的开发流程。

---

# 第一周

---

1. 学习Altium Designer 19软件的使用，完成了FPGA和平衡车拓展板之间的连接板的PCB的设计。
2. 将PCB发出打样，和老师一起设计装配方案，采购必需的元件。
3. 调试摄像头模块，封装并测试了直流电机模块和蓝牙模块的IP核。
4. 搭建安卓开发环境，学习Android Studio的使用，根据蓝牙2.1协议完成了适配HC-06模块的第一版APP的开发。

---

# 第二周

---

1. 根据第一版PCB的问题，纯手工布线设计了第二版PCB，使得PCB的布局更科学，走线更合理。
2. 完成了陀螺仪、超声波模块、电机测速模块的IP封装和NIOS环境下的验证。
3. 在换用BT-05蓝牙模块之后，完成了第二版蓝牙APP，新的APP适配了BT-05所使用的蓝牙4.0 BLE协议和iBeacon协议。

---

# 最后几天

---

1. 将封装好的各子系统的IP核整合到一起，建立总工程，调试各个模块在总的系统下的工作效果，编写顶层逻辑，实现了蓝牙上位机对系统的控制。
2. 根据当前项目的需求在第二版程序的基础上维护了第三版蓝牙上位机APP。

---

- PCB的原理图是由冯湛搏组提供，PCB加工完毕后的装配过程也由冯湛搏组同学完成，这个过程中他们发现的问题帮助我完善除了第二版PCB。
- 蓝牙功能在FPGA上的硬件实现得到了高迎雪组两位同学的帮助。
- 各个模块的调试和整合和平衡车搭建是与宋连涛组、穆帅楠组、陈建川组交流合作完成的。

---

# 课设分工

---

- 薛昕磊：电机驱动模块和电机测速模块的设计、蓝牙模块的设计
- 李英平：连接板PCB设计、陀螺仪模块的封装、模块间的集成调试
- 莫冯然：主导基础实验、摄像头模块、各模块和协议的数据查找
- 武根泽：上位机APP的开发、超声波模块、模块之间的集成调试

---

# 课设成果

---

- 使用FPGA上的NIOS II系统将各传感器连接成一个系统，通过数码管和Console实时显示传感器数据，使用手机APP通过蓝牙控制车轮的运动。

---

#### PCB原理图
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/PicGo20190717145744.jpg)

---

![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/PicGo20190717145810.jpg)

---

![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/PicGo20190717145842.jpg)

---

# 课设感受

---

![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/PicGo20190717145859.png)

---

# 在最后的调试工作中，这是我们做的最多的几件事

- Qsys创建错误、烧录sof文件失败、System ID错误、烧录elf文件失败
- Clean Project -> Generate Project -> Build Project
- 删除Eclipse工程 -> 创建新工程
- 修改RAM大小 -> 重新生成Qsys -> 删除output_files -> 重新生成sof并烧录
- 删除Quartus工程 -> 重建新工程

----

# 谢谢各位老师的指导
