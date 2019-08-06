package is.hail.gateway

import py4j.GatewayServer


object HailJVMEntrypoint {

   final def main(args: Array[String]): Unit = {
     val gatewayServer = new GatewayServer(HailJVMEntrypoint)
     gatewayServer.start()
     println("Started gateway server")

     while (System.in.read() != -1) {
       // Do nothing, just keeping it alive.
     }

     println("System.in connection lost")
     gatewayServer.shutdown()
     System.exit(0)

   }

}