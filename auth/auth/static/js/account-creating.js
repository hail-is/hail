let protocol = location.protocol.replace("http", "ws")
let sock = new WebSocket(protocol + location.host + "{{ base_path }}/creating/wait");
sock.onmessage = function (event) {
    window.location.reload()
}
