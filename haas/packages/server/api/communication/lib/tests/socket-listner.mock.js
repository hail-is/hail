//Communicate with clients, like the website

exports = module.exports = ClientComm;

var socket;

function SocketListener(socketServer)
{
	if(! this instanceof SocketListener)return new SocketListener(socketServer)
	
	this.setListeners;
}

SocketListener.prototype = (function()
{
	assert(this instanceof SocketListener);

})();