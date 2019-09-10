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

     println("Stopping gateway server")
     gatewayServer.shutdown()
     System.exit(0)
   }

}