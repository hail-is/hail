package is.hail.services

import org.json4s.{JInt, JString, JValue}

object UserInfo {
  def fromJValue(jv: JValue): UserInfo = new UserInfo(
    (jv \ "id").asInstanceOf[JInt].num.toInt,
    (jv \ "username").asInstanceOf[JString].s,
    (jv \ "email").asInstanceOf[JString].s
  )
}

class UserInfo(
  val id: Int,
  val username: String,
  val email: String
) {

}
