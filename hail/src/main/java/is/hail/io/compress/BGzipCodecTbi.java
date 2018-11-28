package is.hail.io.compress;

import static htsjdk.tribble.util.TabixUtils.STANDARD_INDEX_EXTENSION;

public class BGzipCodecTbi extends BGzipCodec {
    @Override
    public String getDefaultExtension() {
        return STANDARD_INDEX_EXTENSION;
    }
}
