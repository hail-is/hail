package is.hail.io.avro

import org.json4s.CustomSerializer
import org.json4s.jackson.JsonMethods

import org.apache.avro.Schema

class AvroSchemaSerializer extends CustomSerializer[Schema](_ =>
      (
        { case jv =>
          new Schema.Parser().parse(JsonMethods.compact(jv))
        },
        { case schema: Schema =>
          JsonMethods.parse(schema.toString)
        },
      )
    )
