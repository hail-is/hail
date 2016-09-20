package org.broadinstitute.hail.io.compress;

/**
 * Created by laurent on 9/20/16.
 */
public class BGzipCodecGZ extends BGzipCodec {

    @Override
    public String getDefaultExtension() {
        return ".gz";
    }
}
