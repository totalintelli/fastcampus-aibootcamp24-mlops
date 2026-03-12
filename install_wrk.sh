'''
# 부하 테스트

- 부하 테스트를 통해 API 요청 요구 사항을 만족하는지 확인
'''
#!/bin/bash
git clone https://github.com/wg/wrk.git
cd wrk
make
cp wrk /usr/local/bin
