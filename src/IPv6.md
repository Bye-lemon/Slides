<!-- $theme: default -->

# 当我谈IPv6时，我谈些什么？

#### 李英平


---
# 作为一个文科生，从IPv6的历史谈起。

> *“程序员可能是孤独的，但 IPv6不会孤独。” -- R.L.*

---
## 究竟有多少个IPv
- IPv1: Internet Protocol Version 1
- IPv2: Internet Protocol Version 2
- IPv3: Internet Protocol Version 3
- IPv4: Internet Protocol Version 4
- IPv5: Internet Protocol Version 5
- IPv6: Internet Protocol Version 6
- IPv7: Internet Protocol Version 7
- 
- IPv9: Internet Protocol Version 9
- IPv10: Internet Protocol Version 10

---
# 所以，究竟什么是Internet Protocol

---
- IP是TCP/IP协议族的重要组成，他是一个网络层协议。
- IP的作用是基于IP地址转发信息。

---
# 那么，IPv4和IPv6有什么不同

---
- IPv4的地址是32位的，而IPv6有128位。

  ```
  # IPv4 地址
  192.168.0.100
  # IPv6 地址
  2001:da8:270:2018:f816:3eff:fead:e7e8
  ```
- 这意味着：同等条件下，IPv6有$2^{96}$倍于IPv4的地址个数，它的实际值是：
$$2^{128} = 3.402 \times 10^{38}$$

---
# 所以，这是一个多大的数呢

---
- 假定沙子是一个一毫米的立方体，且地球的陆地表面被沙子覆盖，那么，地球表面的沙子的个数将是：$10^{21}$的数量级。


	> *“使用了 IPv6，连沙子都会有自己的IP地址。” -- 鲁迅*
	
---
## 更多的，IPv6有如下优点
- 在这个物联网时代，IPv6可以让更多的设备联网，符合物联网的主题。
- IPv6速度更快。（尽管这个速度的变化肉眼不可见。）
- 在正常操作的前提下，IPv6拥有更好的安全性。（尽管同样肉眼不可见。）

---
# 那么，我们该如何开发IPv6呢

---
- 既然我们不能解决问题，那最好的办法就是逃避问题。
	
    > *“檀公三十六策，走是上计。” -- 《南齐书·王敬则传》*

---
# 伏笔够了，主角登场：Docker

---
# Docker是什么

---
- Docker是一个使用Go开发的开源的应用容器引擎，是**容器化技术**的代表。
- Docker将应用和所有的依赖打包到一个可以指的容器中，这个容器可以直接运行在任何支持Docker的Linux服务器上。
- 一般地，我们使用Docker完成业务部署，它具有一次编写，随处使用的优点。
- 最后，某种意义上你可以理解成Docker类似于虚拟机。

---
## Docker的架构

![Docker Acchitecture](https://docs.docker.com/engine/images/architecture.svg)

---
# 所以，这和IPv6有什么关系

---
- Docker对外是透明的,我们只要创建一个能在IPv4环境下正常工作的应用层服务，使用Docker把它映射在支持IPv6的Server上，将端口映射，那么服务就可以在IPv6上启动了。
- 从比赛的角度来说，赛尔网络的云主机是经过了IPv6配置的，我们无需进行其他修改，只要安装并启用Docker服务，即可将原有的IPv4服务通过IPv6启动。

---
# 那么，如何使用Docker部署服务

---
## 安装Docker-ce
- 赛尔网络提供的系统是CentOS，CentOS是一个相当保守的Linux发行版，安装Docker需要7.0+的版本。

---
## 编写Dockerfile
```Docker
FROM python
LABEL author="Li Yingping"

RUN apt-get update

ENV PYTHONIOENCODING=utf-8

# Build folder
RUN mkdir -p /deploy/app
WORKDIR /deploy/app
COPY /requirements.txt /deploy/app/requirements.txt
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
```

---
## 使用Docker-compose编排服务
```yaml
version: "3.3"

services:
    webapp:
        build: .
        volumes:
         - ./app:/deploy/app
        ports:
         - "80:5000"
        command: python server.py

```

---
## 更好的使用体验
- 日志记录：Sentry
- CI/CD：Jenkins、Travis、TeamCity
- 集群部署：K8s(Kubernetes)
- ……

---
# EOF