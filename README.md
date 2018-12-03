# NLP Final Project

AWS 账号: yuzhou.mao.work@gmail.com
密码: ASDasd123

CPU server usage:
ssh -i *.pem ubuntu@34.239.156.135

GPU server usage:
ssh -i *.pem ubuntu@35.170.245.144

要在server上用jupyter notebook的话:
ssh -L localhost:9000:localhost:8888 -i *.pem ......
也就是说加上-L localhost:9000:localhost:8888
然后在server里jupyter notebook
并将command line里的host改成9000在浏览器中打开

登入AWS后要使用tensorflow env必须在project folder (nlp-final-autocomplete)输入source .activate
