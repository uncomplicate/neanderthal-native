package uncomplicate.neanderthal;

import static org.junit.Assert.*;
import org.junit.Test;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TestCBLAS {

    @Test
    public void testCBLAS() {
        ByteBuffer x = ByteBuffer.allocateDirect(24);
        x.order(ByteOrder.LITTLE_ENDIAN);
        x.putDouble(0, 1.0);
        x.putDouble(8, 2.0);
        x.putDouble(16, 3.0);
        x.position(0);
        NarSystem.loadLibrary();
        assertEquals(14.0, CBLAS.ddot(3, x, 1, x, 1), 0.0);
    }

}
