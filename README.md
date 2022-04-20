# CryptoDL
거래량이 많이 나오는 인기있는 코인이면서 가격이 적당한 코인 ETC로 학습

## Data
ETC 1분봉 데이터: https://drive.google.com/file/d/1dYu9RcnAdqXFqb6YSDAH33HrzgIKqEo2/view?usp=sharing
학습시 사용하는 데이터는 업로드 중

데이터는 180개씩 묶었으며 Y값은 앞으로 60개의 봉(1시간)동안 close가 구매 가격보다 0.5% 이상인게 있으면 1 아니면 0
구매 가격은 180개씩 묶은 데이터의 마지막 close 값

사용 피쳐
- hour
- minute
- open
- low
- high
- close
- volume
- cci
- mfi10
- fast_k 
- fast_d
- slow_k
- slow_d
- rsi
- ma5
- ma10 
- ma15
- ma20

위 데이터는 각각 업비트, 바이낸스 정보가 같이 존재함
바이낸스는 앞에 b_가 붙음 ex) b_open
