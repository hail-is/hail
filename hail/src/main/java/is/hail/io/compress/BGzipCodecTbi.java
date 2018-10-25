package is.hail.io.compress;

public class BGzipCodecTbi extends BGzipCodec {
    @Override
    public String getDefaultExtension() {
        return ".tbi";
    }
}
