#!/bin/bash
set -e

echo "==============================================================================="
echo "INTEGRATION TESTS (test.sh)"

if [ -t 1 ]; then  # if output is a terminal
    ncolors=$(tput colors)
    if [ -n "$ncolors" ] && [ $ncolors -ge 8 ]  # if we have colors
    then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        NOCOLOR='\033[0m'
    fi
fi

success() {
    printf "${GREEN}success${NOCOLOR}\n"
}
failure() {
    EC=$?
    printf "${RED}failure${NOCOLOR}\n"
    exit $EC
}

echo "Test 1"
build/main > build/out 2> build/err <<EOF
p 10 7
p 63 222
p 10 5
EOF
rm -rf build/expected; touch build/expected
diff build/out build/expected || failure
success

echo "Test 2"
build/main > build/out 2> build/err <<EOF
p 10 7
p 63 222
g 10
g 15
p 15 5
g 15
EOF
cat > build/expected <<EOF
7

5
EOF
diff build/out build/expected || failure
success

echo "Test 3"
build/main > build/out 2> build/err <<EOF
p 10 7
p 13 2
p 17 99
p 12 22
r 10 12
r 10 15
r 14 17
r 0 100
EOF
cat > build/expected <<EOF
10:7
10:7 12:22 13:2

10:7 12:22 13:2 17:99
EOF
diff build/out build/expected || failure
success

echo "Test 4"
build/main > build/out 2> build/err <<EOF
p 10 7
p 12 5
g 10
d 10
g 10
g 12
EOF
cat > build/expected <<EOF
7

5
EOF
diff build/out build/expected || failure
success

echo "Test 5"
build/main > build/out 2> build/err <<EOF
p 10 7
p 12 5
p 13 9
p 33 4
g 10
g 33
D
p 56 80
g 56
D
g 10
EOF
cat > build/expected <<EOF
7
4
10
12
13
33
80
56
7
EOF
diff build/out build/expected || failure
success

echo "Test 6"
build/main > build/out 2> build/err <<EOF
p 1 9
p 3 7
p 4 6
p 2 8
r 1 4
p 5 5
p 6 4
D
p 1 33
r 1 4
r 1 6
EOF
cat > build/expected <<EOF
1:9 2:8 3:7
5
6
1:33 2:8 3:7
1:33 2:8 3:7 4:6 5:5
EOF
diff build/out build/expected || failure
success

echo "Test 7"
build/main > build/out 2> build/err <<EOF
p 1 9
p 3 7
p 4 6
p 2 8
r 1 4
p 5 5
p 6 4
p 7 3
p 2 99
D
p 8 2
r 1 4
r 1 6
EOF
cat > build/expected <<EOF
1:9 2:8 3:7
2
5
6
7
1:9 2:99 3:7
1:9 2:99 3:7 4:6 5:5
EOF
diff build/out build/expected || failure
success

echo "Test 8"
build/main > build/out 2> build/err <<EOF
p 1 3
p 10 1
p 11 1
p 12 1
d 1
g 1
EOF
cat > build/expected <<EOF

EOF
diff build/out build/expected || failure
success

echo "Test 9"
build/main > build/out 2> build/err <<EOF
p 1 3
p 10 1
p 11 1
p 12 1
d 1
p 13 1
p 14 1
p 15 1
p 16 1
g 1
EOF
cat > build/expected <<EOF

EOF
diff build/out build/expected || failure
success

echo "Test 10"
build/main > build/out 2> build/err <<EOF
p 1 3
p 10 1
p 11 1
p 12 1
d 1
p 13 1
p 14 1
p 15 1
p 16 1
r 0 9
EOF
cat > build/expected <<EOF

EOF
diff build/out build/expected || failure
success

echo "Test 11"
build/main > build/out 2> build/err <<EOF
p 1 3
p 10 1
p 11 1
p 12 1
d 1
r 0 9
EOF
cat > build/expected <<EOF

EOF
diff build/out build/expected || failure
success

echo "Test 12"
build/main > build/out 2> build/err <<EOF
p 1 3
d 1
r 0 9
EOF
cat > build/expected <<EOF

EOF
diff build/out build/expected || failure
success

echo "Test 13"
build/main > build/out 2> build/err <<EOF
p 1 11
p 2 22
p 7 77
p 3 33
p 8 88
p 4 44
p 5 55
p 9 99
p 6 66
p 10 100
p 11 110
p 13 130
p 12 120
p 17 170
p 14 140
p 16 160
p 15 150
p 18 180
p 19 190
p 20 200
p 22 220
p 23 230
p 21 210
p 24 240
p 25 250
p 26 260
p 28 280
p 27 270
p 29 290
p 32 320
p 30 300
p 31 310
g 30
g 22
g 11
g 8
EOF
cat > build/expected <<EOF
300
220
110
88
EOF
diff build/out build/expected || failure
success

echo "Test 14"
build/main > build/out 2> build/err <<EOF
p 1 11
p 2 22
p 7 77
p 3 33
p 6 66
p 10 100
p 11 110
p 13 130
p 12 120
p 17 170
p 14 140
p 16 160
p 15 150
p 18 180
p 19 190
p 20 200
p 22 220
p 8 88
p 4 44
p 5 55
p 9 99
p 23 230
p 21 210
p 24 240
p 25 250
p 26 260
p 28 280
p 27 270
p 29 290
p 32 320
p 30 300
p 31 310
r 3 13
r 31 35
r 33 80
EOF
cat > build/expected <<EOF
3:33 4:44 5:55 6:66 7:77 8:88 9:99 10:100 11:110 12:120
31:310 32:320

EOF
diff build/out build/expected || failure
success

echo "Test 15"
build/main > build/out 2> build/err <<EOF
p 1 11
p 2 22
p 7 77
p 3 33
p 8 88
p 4 44
p 5 55
p 9 99
p 6 66
p 10 100
p 11 110
p 13 130
d 10
p 12 120
p 17 170
p 14 140
p 16 160
p 5 150
p 18 180
p 19 190
p 20 200
p 22 220
p 23 230
p 21 210
p 24 240
p 25 250
p 26 260
p 28 280
p 27 270
p 29 290
p 32 320
p 30 300
p 1 310
g 1
g 5
g 10
g 8
d 8
g 8
g 31
r 1 11
EOF
cat > build/expected <<EOF
310
150

88


1:310 2:22 3:33 4:44 5:150 6:66 7:77 9:99
EOF
diff build/out build/expected || failure
success

echo "Test 16"
build/main > build/out 2> build/err <<EOF
p 1 11
p 2 22
p 7 77
p 3 33
p 8 88
p 4 44
p 5 55
p 9 99
p 6 66
d 8
p 10 100
p 11 110
p 13 130
p 12 120
p 17 170
p 14 140
p 16 160
p 5 150
p 18 180
d 2
p 19 190
p 20 200
p 22 220
p 23 230
p 21 210
p 24 240
p 25 250
p 26 260
p 28 280
p 27 270
d 10
d 3
p 29 290
p 32 320
p 30 300
p 1 310
g 1
g 5
g 10
g 8
g 31
g 4
g 3
g 2
r 1 11
EOF
cat > build/expected <<EOF
310
150



44


1:310 4:44 5:150 6:66 7:77 9:99
EOF
diff build/out build/expected || failure
success
