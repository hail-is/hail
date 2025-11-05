const basePath = document.getElementById('_base_path').value;
let protocol = location.protocol.replace("http", "ws")
let sock = new WebSocket(protocol + location.host + `${basePath}/creating/wait`);
sock.onmessage = function (event) {
    window.location.reload()
}
