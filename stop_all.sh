#!/bin/bash

# 关闭所有jina_server.py相关进程
ps aux | grep jina_server.py | grep -v grep | awk '{print $2}' | xargs kill -9


