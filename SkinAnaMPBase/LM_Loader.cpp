//
//  Utils.cpp

/*******************************************************************************
Author: Fu Xiaoqiang
Date:   2022/9/15

********************************************************************************/

#include "LM_Loader.hpp"


//-------------------------------------------------------------------------------------------

/******************************************************************************************

 ******************************************************************************************/

int general_lm[478][2] = {
    { 1139 , 2640 },
    { 1109 , 2379 },
    { 1119 , 2404 },
    { 1049 , 2064 },
    { 1108 , 2307 },
    { 1113 , 2180 },
    { 1131 , 1823 },
    { 523 , 1679 },
    { 1142 , 1533 },
    { 1146 , 1410 },
    { 1163 , 934 },
    { 1140 , 2658 },
    { 1141 , 2675 },
    { 1143 , 2685 },
    { 1144 , 2683 },
    { 1145 , 2714 },
    { 1145 , 2749 },
    { 1146 , 2778 },
    { 1132 , 2767 },
    { 1112 , 2399 },
    { 1032 , 2371 },
    { 319 , 1147 },
    { 803 , 1757 },
    { 714 , 1764 },
    { 627 , 1755 },
    { 504 , 1690 },
    { 870 , 1737 },
    { 653 , 1507 },
    { 752 , 1528 },
    { 563 , 1509 },
    { 508 , 1536 },
    { 465 , 1720 },
    { 874 , 2789 },
    { 492 , 1659 },
    { 296 , 1517 },
    { 401 , 1623 },
    { 739 , 2132 },
    { 1049 , 2620 },
    { 1060 , 2663 },
    { 978 , 2617 },
    { 930 , 2612 },
    { 997 , 2649 },
    { 950 , 2635 },
    { 815 , 2588 },
    { 1044 , 2369 },
    { 1030 , 2298 },
    { 404 , 1398 },
    { 896 , 1931 },
    { 870 , 2261 },
    { 869 , 2205 },
    { 514 , 2065 },
    { 1038 , 2178 },
    { 559 , 1342 },
    { 459 , 1352 },
    { 367 , 1042 },
    { 927 , 1466 },
    { 836 , 1571 },
    { 758 , 2526 },
    { 443 , 2171 },
    { 928 , 2312 },
    { 988 , 2348 },
    { 855 , 2602 },
    { 871 , 2611 },
    { 425 , 1275 },
    { 874 , 2278 },
    { 707 , 1370 },
    { 692 , 1291 },
    { 658 , 933 },
    { 404 , 1176 },
    { 695 , 1141 },
    { 371 , 1337 },
    { 348 , 1255 },
    { 1053 , 2644 },
    { 987 , 2633 },
    { 939 , 2623 },
    { 945 , 2325 },
    { 863 , 2607 },
    { 894 , 2636 },
    { 879 , 2614 },
    { 959 , 2321 },
    { 959 , 2644 },
    { 1007 , 2660 },
    { 1069 , 2676 },
    { 1038 , 2752 },
    { 1053 , 2766 },
    { 1054 , 2737 },
    { 1060 , 2703 },
    { 1070 , 2673 },
    { 959 , 2643 },
    { 944 , 2654 },
    { 929 , 2668 },
    { 919 , 2680 },
    { 826 , 2448 },
    { 350 , 1767 },
    { 1115 , 2402 },
    { 922 , 2630 },
    { 907 , 2633 },
    { 1018 , 2384 },
    { 899 , 2316 },
    { 1003 , 2367 },
    { 822 , 1961 },
    { 692 , 2005 },
    { 863 , 2224 },
    { 469 , 968 },
    { 510 , 1132 },
    { 532 , 1259 },
    { 873 , 2648 },
    { 898 , 1343 },
    { 904 , 1167 },
    { 884 , 924 },
    { 548 , 1728 },
    { 401 , 1757 },
    { 907 , 1719 },
    { 445 , 1542 },
    { 947 , 1896 },
    { 908 , 2264 },
    { 339 , 1755 },
    { 435 , 1836 },
    { 525 , 1892 },
    { 682 , 1901 },
    { 799 , 1879 },
    { 879 , 1851 },
    { 1052 , 1831 },
    { 358 , 1938 },
    { 399 , 1501 },
    { 1071 , 2394 },
    { 901 , 2036 },
    { 307 , 1425 },
    { 938 , 1820 },
    { 862 , 2206 },
    { 472 , 1630 },
    { 908 , 2194 },
    { 385 , 1960 },
    { 904 , 1695 },
    { 969 , 2180 },
    { 584 , 2509 },
    { 603 , 2524 },
    { 319 , 1857 },
    { 491 , 2369 },
    { 318 , 1372 },
    { 866 , 2831 },
    { 1080 , 2396 },
    { 836 , 2078 },
    { 361 , 1613 },
    { 625 , 1712 },
    { 710 , 1719 },
    { 883 , 2638 },
    { 390 , 2087 },
    { 991 , 2924 },
    { 798 , 2759 },
    { 713 , 2660 },
    { 1153 , 1185 },
    { 1135 , 2955 },
    { 787 , 1713 },
    { 852 , 1698 },
    { 889 , 1693 },
    { 362 , 1456 },
    { 830 , 1700 },
    { 752 , 1715 },
    { 671 , 1719 },
    { 592 , 1707 },
    { 543 , 1690 },
    { 300 , 1268 },
    { 564 , 1696 },
    { 1123 , 2452 },
    { 888 , 2430 },
    { 925 , 2308 },
    { 1017 , 2445 },
    { 1138 , 1664 },
    { 684 , 2630 },
    { 773 , 2727 },
    { 982 , 2914 },
    { 516 , 2365 },
    { 879 , 1695 },
    { 994 , 1976 },
    { 1133 , 2947 },
    { 887 , 2854 },
    { 357 , 2027 },
    { 1008 , 2659 },
    { 994 , 2679 },
    { 983 , 2704 },
    { 977 , 2730 },
    { 946 , 2707 },
    { 911 , 2622 },
    { 899 , 2614 },
    { 890 , 2606 },
    { 782 , 2483 },
    { 478 , 2194 },
    { 1000 , 1860 },
    { 952 , 1633 },
    { 910 , 1640 },
    { 923 , 2631 },
    { 500 , 2331 },
    { 1014 , 1647 },
    { 905 , 2744 },
    { 1119 , 2063 },
    { 1049 , 1956 },
    { 1125 , 1951 },
    { 936 , 2100 },
    { 1131 , 2902 },
    { 1132 , 2831 },
    { 1014 , 2809 },
    { 742 , 2589 },
    { 798 , 2253 },
    { 817 , 2669 },
    { 622 , 2206 },
    { 735 , 2320 },
    { 575 , 2301 },
    { 990 , 2872 },
    { 898 , 2119 },
    { 689 , 2601 },
    { 781 , 2696 },
    { 677 , 2502 },
    { 429 , 2206 },
    { 589 , 2461 },
    { 408 , 2192 },
    { 692 , 2401 },
    { 947 , 2003 },
    { 933 , 2308 },
    { 893 , 2295 },
    { 964 , 2281 },
    { 877 , 1537 },
    { 734 , 1465 },
    { 611 , 1442 },
    { 518 , 1448 },
    { 461 , 1479 },
    { 446 , 1624 },
    { 298 , 1687 },
    { 500 , 1770 },
    { 586 , 1812 },
    { 699 , 1824 },
    { 804 , 1812 },
    { 881 , 1788 },
    { 929 , 1769 },
    { 330 , 1589 },
    { 899 , 2303 },
    { 987 , 2077 },
    { 988 , 2340 },
    { 1028 , 2368 },
    { 989 , 2338 },
    { 919 , 2322 },
    { 1045 , 2384 },
    { 1053 , 2386 },
    { 931 , 1696 },
    { 967 , 1729 },
    { 983 , 1761 },
    { 513 , 1674 },
    { 475 , 1570 },
    { 1191 , 2069 },
    { 1773 , 1739 },
    { 1201 , 2377 },
    { 2082 , 1206 },
    { 1495 , 1796 },
    { 1582 , 1809 },
    { 1670 , 1804 },
    { 1800 , 1746 },
    { 1427 , 1769 },
    { 1655 , 1565 },
    { 1556 , 1575 },
    { 1745 , 1574 },
    { 1802 , 1602 },
    { 1851 , 1770 },
    { 1406 , 2812 },
    { 1809 , 1718 },
    { 2085 , 1583 },
    { 1943 , 1673 },
    { 1542 , 2158 },
    { 1231 , 2619 },
    { 1226 , 2662 },
    { 1314 , 2614 },
    { 1373 , 2606 },
    { 1298 , 2645 },
    { 1355 , 2629 },
    { 1483 , 2608 },
    { 1176 , 2374 },
    { 1189 , 2303 },
    { 1929 , 1466 },
    { 1387 , 1947 },
    { 1381 , 2278 },
    { 1388 , 2221 },
    { 1785 , 2108 },
    { 1191 , 2183 },
    { 1750 , 1404 },
    { 1860 , 1422 },
    { 2018 , 1097 },
    { 1354 , 1484 },
    { 1469 , 1609 },
    { 1541 , 2548 },
    { 1922 , 2224 },
    { 1322 , 2325 },
    { 1256 , 2356 },
    { 1468 , 2591 },
    { 1450 , 2600 },
    { 1911 , 1346 },
    { 1383 , 2294 },
    { 1591 , 1413 },
    { 1608 , 1339 },
    { 1690 , 967 },
    { 1956 , 1236 },
    { 1630 , 1176 },
    { 1979 , 1402 },
    { 2027 , 1320 },
    { 1230 , 2643 },
    { 1307 , 2630 },
    { 1365 , 2617 },
    { 1303 , 2336 },
    { 1458 , 2596 },
    { 1422 , 2628 },
    { 1442 , 2604 },
    { 1279 , 2332 },
    { 1348 , 2638 },
    { 1290 , 2657 },
    { 1221 , 2674 },
    { 1233 , 2760 },
    { 1241 , 2766 },
    { 1238 , 2737 },
    { 1231 , 2702 },
    { 1223 , 2672 },
    { 1349 , 2638 },
    { 1364 , 2649 },
    { 1379 , 2662 },
    { 1390 , 2675 },
    { 1452 , 2466 },
    { 2046 , 1827 },
    { 1393 , 2622 },
    { 1408 , 2625 },
    { 1226 , 2390 },
    { 1359 , 2330 },
    { 1241 , 2374 },
    { 1466 , 1982 },
    { 1599 , 2035 },
    { 1399 , 2240 },
    { 1897 , 1015 },
    { 1833 , 1183 },
    { 1786 , 1324 },
    { 1416 , 2666 },
    { 1383 , 1367 },
    { 1409 , 1186 },
    { 1452 , 942 },
    { 1751 , 1782 },
    { 1930 , 1804 },
    { 1388 , 1748 },
    { 1880 , 1605 },
    { 1332 , 1909 },
    { 1332 , 2279 },
    { 2002 , 1815 },
    { 1870 , 1884 },
    { 1760 , 1935 },
    { 1598 , 1934 },
    { 1481 , 1904 },
    { 1397 , 1873 },
    { 1213 , 1836 },
    { 1969 , 1995 },
    { 1942 , 1560 },
    { 1154 , 2396 },
    { 1374 , 2051 },
    { 2107 , 1486 },
    { 1336 , 1839 },
    { 1408 , 2222 },
    { 1841 , 1690 },
    { 1340 , 2208 },
    { 1995 , 2018 },
    { 1395 , 1725 },
    { 1268 , 2190 },
    { 1735 , 2553 },
    { 1726 , 2567 },
    { 2039 , 1919 },
    { 1837 , 2417 },
    { 2062 , 1437 },
    { 1415 , 2857 },
    { 1150 , 2398 },
    { 1442 , 2097 },
    { 2008 , 1661 },
    { 1670 , 1771 },
    { 1587 , 1773 },
    { 1434 , 2630 },
    { 1933 , 2141 },
    { 1284 , 2939 },
    { 1503 , 2790 },
    { 1603 , 2698 },
    { 1513 , 1759 },
    { 1448 , 1736 },
    { 1411 , 1727 },
    { 1999 , 1511 },
    { 1469 , 1729 },
    { 1547 , 1748 },
    { 1625 , 1759 },
    { 1701 , 1753 },
    { 1751 , 1741 },
    { 2110 , 1329 },
    { 1730 , 1757 },
    { 1379 , 2443 },
    { 1321 , 2322 },
    { 1235 , 2452 },
    { 1624 , 2668 },
    { 1523 , 2761 },
    { 1287 , 2929 },
    { 1830 , 2412 },
    { 1419 , 1723 },
    { 1266 , 1985 },
    { 1400 , 2878 },
    { 1991 , 2085 },
    { 1291 , 2656 },
    { 1306 , 2677 },
    { 1317 , 2702 },
    { 1324 , 2728 },
    { 1333 , 2720 },
    { 1403 , 2613 },
    { 1414 , 2605 },
    { 1422 , 2597 },
    { 1507 , 2503 },
    { 1828 , 2240 },
    { 1273 , 1869 },
    { 1340 , 1656 },
    { 1390 , 1671 },
    { 1393 , 2622 },
    { 1815 , 2376 },
    { 1258 , 1664 },
    { 1375 , 2762 },
    { 1203 , 1961 },
    { 1322 , 2112 },
    { 1254 , 2818 },
    { 1554 , 2617 },
    { 1477 , 2273 },
    { 1474 , 2693 },
    { 1665 , 2241 },
    { 1547 , 2345 },
    { 1721 , 2339 },
    { 1276 , 2884 },
    { 1369 , 2133 },
    { 1613 , 2637 },
    { 1512 , 2726 },
    { 1621 , 2533 },
    { 1895 , 2257 },
    { 1716 , 2500 },
    { 1931 , 2246 },
    { 1599 , 2428 },
    { 1321 , 2016 },
    { 1304 , 2321 },
    { 1355 , 2311 },
    { 1264 , 2291 },
    { 1421 , 1567 },
    { 1569 , 1513 },
    { 1694 , 1504 },
    { 1792 , 1517 },
    { 1855 , 1548 },
    { 1883 , 1681 },
    { 2072 , 1752 },
    { 1803 , 1820 },
    { 1709 , 1855 },
    { 1592 , 1862 },
    { 1486 , 1842 },
    { 1409 , 1813 },
    { 1358 , 1791 },
    { 2079 , 1649 },
    { 1353 , 2318 },
    { 1260 , 2086 },
    { 1239 , 2350 },
    { 1203 , 2375 },
    { 1244 , 2349 },
    { 1334 , 2334 },
    { 1182 , 2389 },
    { 1179 , 2390 },
    { 1365 , 1724 },
    { 1319 , 1750 },
    { 1289 , 1780 },
    { 1784 , 1729 },
    { 1838 , 1634 },
    { 692 , 1708 },
    { 766 , 1714 },
    { 693 , 1641 },
    { 618 , 1702 },
    { 691 , 1775 },
    { 1588 , 1745 },
    { 1668 , 1747 },
    { 1590 , 1674 },
    { 1509 , 1742 },
    { 1587 , 1814 }
};

void FillData(FaceInfo& faceInfo)
{
    for(int i = 0; i<468; i++)
    {
        faceInfo.lm_2d[i][0] = general_lm[i][0];
        faceInfo.lm_2d[i][1] = general_lm[i][1];
    }
}

//-------------------------------------------------------------------------------------------
