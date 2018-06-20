#Computer Organization and Design: The Hardware/Software Interface

## Preface

### About This Book

儘管程序員可以忽略這些建議，並依靠計算機架構師，編譯器編寫人員和芯片工程師來使他們的程序運行得更快或是更節能，但那個時代已經結束.

>While programmers could ignore the advice and rely on computer architects, compiler writers, and silicon engineers to make their programs run faster or be more energy-effi cient without change, that era is over.


我們的觀點是，至少在未來十年，如果程序員希望程序在並行計算機上高效運行，大多數程序員將不得不了解硬件/軟件接口, 我們的觀點是，至少在未來的十年中，大多數程序員將不得不了解硬件/軟件不僅界面，如果他們想的方案在並行計算機上高效運行.

>Our view is that for at least the next decade, most programmers are going to have to understand the hardware/soft ware interface if they want programs to run efficiently on parallel computers.

## 1.Computer Abstractions and Technology

### 1.1 introduction

介紹電腦


* personal computer (PC)
* Servers
* supercomputer
* embedded computer
* Personal mobile devices (PMDs)
* Cloud Computing
* Software as a Service (SaaS)

單位

Decimal term|Abbreviation|Value|Binary term|Abbreviation|Value
--|--|--|--|--|--
kilobyte|KB|$10^3$|kibibyte|KiB|$2^{10}$
megabyte|MB|$10^6$|mebibyte|MiB|$2^{20}$
gigabyte|GB|$10^9$|gibibyte|GiB|$2^{30}$

!!! question "iphone 16GB 為什麼被消費者誤認14"GB"? " 
    因為廣告iphone 16 GB ,但手機系統顯示的單位是Gib</br>
    根據換算</br>
    16GB=16*$10^9$Bytes</br>
    $16*10^9$ Bytes =$\frac{16*10^9}{2^{30}}$ GiB=14.9 GiB</br>
    14GiB就是這樣來的</br>
    所以單位要了解清楚才不會產生誤會
    