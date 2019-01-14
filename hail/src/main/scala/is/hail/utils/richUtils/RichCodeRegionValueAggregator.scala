package is.hail.utils.richUtils

import is.hail.annotations.RegionValueBuilder
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.asm4s.Code

class RichCodeRegionValueAggregator(val rva: Code[RegionValueAggregator]) {
  def result(rvb: Code[RegionValueBuilder]): Code[Unit] = {
    rva.invoke[RegionValueBuilder, Unit]("result", rvb)
  }
}
