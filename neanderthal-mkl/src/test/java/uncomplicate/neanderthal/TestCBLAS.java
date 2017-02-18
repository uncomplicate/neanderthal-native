//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

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
        assertEquals(14.0, CBLAS.ddot(3, x, 0, 1, x, 0, 1), 0.0);
    }

}
