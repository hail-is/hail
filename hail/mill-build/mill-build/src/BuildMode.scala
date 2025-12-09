package millbuild

import upickle.default.ReadWriter

enum BuildMode derives ReadWriter:
  case Dev, CI, Release