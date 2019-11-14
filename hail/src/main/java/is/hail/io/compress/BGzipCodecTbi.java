package is.hail.io.compress;

import static htsjdk.samtools.util.FileExtensions.TABIX_INDEX;

public class BGzipCodecTbi extends BGzipCodec {
    @Override
    public String getDefaultExtension() {
        return TABIX_INDEX;
    }
}
