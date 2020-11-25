#!/bin/bash
set -e

echo 1
build/main > build/out 2> build/err <<EOF
p 10 7
p 63 222
p 10 5
EOF
rm -rf build/expected; touch build/expected
diff build/out build/expected
echo success

echo 2
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
diff build/out build/expected
echo success

echo 3
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
diff build/out build/expected
echo success

echo 4
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
diff build/out build/expected
echo success
