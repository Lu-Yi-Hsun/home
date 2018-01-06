# Block Chain

`block chain on bitcoin`

!!! question "既然選更長的分支，那我用很低的難度去求解怎麼辦?"
    客戶端在眾多分支中找到符合當前難度且最長的。
    
[View last Block](https://blockexplorer.com/)

## 欄位介紹

[sample raw data web](https://blockexplorer.com/block/000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26)

[sample raw data json](https://blockexplorer.com/api/block/000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26)

```json
{
"hash":"000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26",
"size":986803,
"height":494333,
"version":536870912,
"merkleroot":"eb60eac626c77e2719621f3d2bc5379f43f5a62b5994e184520eb290bb31960c",
"tx":["..........."],
"time":1510663787,
"nonce":3267589382,
"bits":"1800ce4b",
"difficulty":1364422081125.1475,
"chainwork":"000000000000000000000000000000000000000000b09e2108dc9f58fa17f83e",
"confirmations":9,
"previousblockhash":"0000000000000000001072fdaa28f5b128009e8580f6080ca82063f9a912cbbc","nextblockhash":"0000000000000000000960da72f18edad9d101fb5c3a2ac1ecbfea7600d82575",
"reward":12.5,
"isMainChain":true,
"poolInfo":{"poolName":"AntMiner","url":"https://bitmaintech.com/"}}
```

---

### hash

!!! note "Block hashing algorithm"
    hash256(hash256(Version+hashPrevBlock+hashMerkleRoot+Time+Bits+Nonce))

|Field|Purpose|Updated when...|Size (Bytes)|
|---|---|---|---|
|Version|Block version number|You upgrade the software and it specifies a new version|4|
|hashPrevBlock|256-bit hash of the previous block header|A new block comes in|32|
|hashMerkleRoot|256-bit hash based on all of the transactions in the block|A transaction is accepted|32|
|Time|Current timestamp as seconds since 1970-01-01T00:00 UTC|Every few seconds|4|
|Bits|Current target in compact format|The difficulty is adjusted|4|
|Nonce|32-bit number (starts at 0)|A hash is tried (increments)|4|
https://en.bitcoin.it/wiki/Block_hashing_algorithm

!!! warning
    The output of blockexplorer displays the hash values as big-endian numbers</br>
    but python3 hashlib is little end

??? note "CODE on Python3"
    ``` python
    '''
    source
    support blockexplorer & blockchain
    https://blockexplorer.com/api/block/000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26
    or
    https://blockchain.info/rawblock/000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26
    '''
    import hashlib
    import codecs
    import struct

    '''
    import hashlib using "little-end" but blockexplorer is "biggest-end" ,we need convert biggest-end to little-end
    '''
    def btol(st):
        if len(st)%2 !=0:
            return "error need 2 pair"
        else:
            ans=""
            for i in range(int(len(st)/2)):
                index=i*2
                ans=st[index]+st[index+1]+ans
            return ans
    ''' we can get sample on blockexplorer '''    
    Version="536870912" #4bytes so need 8 hex
    hashPrevBlock="0000000000000000001072fdaa28f5b128009e8580f6080ca82063f9a912cbbc"#32bytes
    hashMerkleRoot="eb60eac626c77e2719621f3d2bc5379f43f5a62b5994e184520eb290bb31960c"#32bytes
    Time="1510663787"#4bytes
    Bits="1800ce4b"#4bytes
    Nonce="3267589382"#4bytes
    
    control=input("blockexplorer.com Data input \"1\" blockchain.info Date input \"2\" \n")

    if control.isdecimal():
        if int(control)==1:
            pass
        elif int(control)==2:
            Bits=str(hex(int(Bits))[2:]).zfill(8)
        else:
            print("error please input 1 or 2")
            exit(1)
    else:
        print("error please input 1 or 2")
        exit(1)
        
    header_hex = (
    btol(str(hex(int(Version))[2:]).zfill(8)) +
    btol(hashPrevBlock) +
    btol(hashMerkleRoot) +
    btol(str(hex(int(Time))[2:]).zfill(8)) +
    btol(Bits) +
    btol(str(hex(int(Nonce))[2:])).zfill(8))

    

    header_bin = codecs.decode(header_hex, "hex")
    hash = hashlib.sha256(hashlib.sha256(header_bin).digest()).digest()


    '''we want to view biggest-end'''
    print("view hash by big-endian")
    print (codecs.encode(hash[::-1], "hex"))
    ```
---


### size

Size (bytes)

---

### height

代表前面有多少塊block chain ,創世第一個blockchain的hight值為零

---

### version
Block version number	You upgrade the software and it specifies a new version	4bytes

---

### merkleroot

![](/images/blockchain/mark.png)

$H_{A} = SHA_{256}(SHA_{256}(Tx A))$

$H_{AB} = SHA_{256}(SHA_{256}(H_{A}+H_{B}))$


### tx(Transactions)
交易訊息
![](/images/blockchain/tx.png)


---

### time

Current timestamp as seconds since 1970-01-01T00:00 UTC

---

### nonce

控制碼藉由修改這裡的數值來讓hash變動
!!! Warning 
    nonce只有4bytes並不代表只需要$2^{32}$就可以破解sha256</br>
    有可能算完還找不到答案,就必須更改其他欄位好讓hash符合規定
    

---

### bits
bits to target

#### solution1

!!! note "bits mean"
    like 0x1d00ffff</br>
    all large is 32 bytes</br>
    * 0x$\color{blue}{1d}$ --- $(1d)_{16}=(26)_{10}$ so we know 26 bytes after 00ffff </br>
    * 0x$\color{red}{00ffff}$ --- target prefix</br>
    $000000\color{red}{00ffff}\color{blue}{0000000000000000000000000000000000000000000000000000}$

 
 
#### solution2 

better performence
```
exponent=bits[0~1]

coefficient=bits[2~7]

target = coefficient * 2^(8 * (exponent – 3))
```

!!! note "CODE on Python3"
    bite to target</br>
    solution1
    ```
    bits="1d00ffff"
    print(bits[2:].ljust(2*int(bits[0:2],16),'0').zfill(64))
    ```
    solution2
    ```
    print(hex(0x00ffff*2**(8*(0x1d - 3)))[2:].zfill(64))
    ```



---
 
### difficulty

!!! note "公式"
    兩種difficulty 一般使用bdiff</br>
    `bdiff 定義:1困難度（difficulty）的bits=0x1D00FFFF`  </br>
    `pdiff-target:0x00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF`</br>
    $1=\frac{f_{conv}(0x1D00FFFF)}{f_{conv}(0x1D00FFFF)}$</br>
    exponent=bits[0-1]</br>
    coefficient=bits[2-7]</br>  
    $f_{conv}(bits)$=coefficient * 2^(8 * (exponent – 3))</br>
    $f_{conv}(\color{red}{1D}\color{blue}{00FFFF})=\color{blue}{00FFFF}_{16} * 2^{( 8 *( \color{red}{1D}_{16}  -  3 ))}$ </br>
    $difficulty=\frac{f_{conv}(0x1D00FFFF)}{f_{conv}(bits) }$</br>
   
    
!!!info "target mean"
    $target=f_{conv}(bits)$</br>
    target意思就是在挖礦的時候nonce該怎麼調整好讓hash<target </br>
    target 越小代表難度越大
    



!!! question "we know bits=1800ce4b difficulty(bdiff)=?"
    ``` python
    difficulty=0x00ffff*2**(8*(0x1d - 3))/float(0x00ce4b*2**(8*(0x18 - 3)))
    ```
    Answer:difficulty=1364422081125.1475 

[Bitcoin Difficulty](https://bitcoinwisdom.com/bitcoin/difficulty)

---

### chainwork

十六進制儲存 當前鏈所需的總Hash
!!! note "範例"
    set chainwork=10Hash
    你的電腦1H/s 
    代表你要算玩這串區塊練區要10秒


---
### confirmations

[Confirmation](https://en.bitcoin.it/wiki/Confirmation)
代表該blockchain的版本之下 前面有多少塊block
一般來說等待6塊就可以確認交易

---

### previousblockhash

前一個區塊鏈的HASH

---

### reward

算出該區塊練所給的獎勵 </br>
2009~2012 50BTC</br>
2013~2016 25BTC</br>
2017~2020 12.5BTC</br>
2020~     6.25BTC </br>

---

 
### isMainChain

是不是最長的那個區塊練

### poolInfo

礦場資訊
 
---


## mining

example "bits":"1800ce4b"

$target=(0x00ce4b)_{16} * 2^{(8 * ((0x18)_{16} – 3))}$

!!! note 
    我們想要藉由更動nonce 來讓hash<target


target=`000000000000000000CE4B000000000000000000000000000000000000000000`                                                          
hash=  `000000000000000000011f35a721c6065b447eef96640ce0ca9ba9a98edd9a26`
                        
## Hash Rate

!!! note "計算駭客要攻擊51%的難度"
    [硬體參考數據](https://bitcointalk.org/index.php?topic=1808668.40)</br>
    [Hashrate](https://bitcoinwisdom.com/bitcoin/difficulty)</br>
    注意：該換算只能參考用</br>
    Intel Xeon E5-2698 V4 $\simeq$ 800H/s</br>
    Intel Xeon Phi 7210 $\simeq$ 600H/s</br>
    [天和二號](https://zh.wikipedia.org/wiki/%E5%A4%A9%E6%B2%B3%E4%BA%8C%E5%8F%B7)</br>
    32,000顆Xeon E5主處理器和48,000個Xeon Phi協處理器</br>
    天和二號:800*32,000+48,000*600=54400000H/s=54400000MH/s=54400GH/s</br>
    全球比特幣Hashrate：7602699877 GH/s</br>
    全球比特幣/天和二號=139755</br>
    結論需要13/2萬組天和二號超級電腦才能超過51%全世界的電腦</br>
    由於目前比特幣是使用ASIC所以效能才會如此之大是cpu所無法批擬</br>

256/8
32bytes