package is.hail.nativecode

import is.hail.utils

// Support for calls from C++ to a limited set of Scala and Java classes/methods
// src/main/c/Upcalls.cpp will create one instance of this class

object Upcalls {
  var testMsg = ""
}

class Upcalls {
  // simple upcall for testing
  def setTestMsg(msg: String): Unit = {
    System.err.println(s"DEBUG: setTestMsg(${msg})")
    Upcalls.testMsg = msg
  }

  // Logging
  def info(msg: String): Unit = utils.info(msg)
  
  def warn(msg: String): Unit = utils.warn(msg)
  
  def error(msg: String): Unit = utils.error(msg)

}
