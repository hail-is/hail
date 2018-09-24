uname -a
mkdir -p artifacts/foo/bar
cat <<EOF > artifacts/index.html
<html>
<body>
<h1>Hello World!</h1>
<a href='foo/bar/thing'>foo/bar/thing!</a>
</body>
</html>
EOF
cat <<EOF > artifacts/foo/bar/thing
this is a thing
thing thing thing
thing a ding
EOF
