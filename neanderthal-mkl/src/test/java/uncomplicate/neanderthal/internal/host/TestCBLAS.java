//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.host;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.BeforeClass;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TestCBLAS {

    @Test
    public void testCBLASddot() {
        ByteBuffer x = ByteBuffer.allocateDirect(24);
        x.order(ByteOrder.LITTLE_ENDIAN);
        x.putDouble(0, 1.0);
        x.putDouble(8, 2.0);
        x.putDouble(16, 3.0);
        x.position(0);
        assertEquals(14.0, CBLAS.ddot(3, x, 0, 1, x, 0, 1), 0.0);
    }

    @Test
    public void testCBLASdcopy() {
        ByteBuffer x = ByteBuffer.allocateDirect(24);
        x.order(ByteOrder.LITTLE_ENDIAN);
        x.putDouble(0, 1.0);
        x.putDouble(8, 2.0);
        x.putDouble(16, 3.0);
        x.position(0);
        ByteBuffer y = ByteBuffer.allocateDirect(24);
        y.order(ByteOrder.LITTLE_ENDIAN);
        CBLAS.dcopy(3, x, 0, 1, y, 0, 1);
        assertEquals(1.0, y.getDouble(0), 0.0);
    }

    /* custom xerbla currently does not work
    @Test(expected = RuntimeException.class)
    public void testXerbla() {
        ByteBuffer a = ByteBuffer.allocateDirect(24);
        ByteBuffer x = ByteBuffer.allocateDirect(24);
        ByteBuffer y = ByteBuffer.allocateDirect(8);
        CBLAS.dgemv(-11, -12, 1, 3, 5.0, a, 0, 1, x, 0, 1, 3.0, y, 0, 1);
    }    */
}
