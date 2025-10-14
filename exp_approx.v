// exp_approx.v
`include "fixed_pkg.v"
module exp_approx (
    input  wire signed [`QWIDTH-1:0] y_in, // Q16.16, must be >=0
    output wire signed [`QWIDTH-1:0] y_out // Q16.16 approx exp(-y_in)
);
    // y^2 (Q32.32), y^3 (Q48.48) etc.
    wire signed [63:0] y2 = $signed(y_in) * $signed(y_in);
    wire signed [95:0] y3 = $signed(y2) * $signed(y_in);

    // terms: careful with shifts: y2 >> Q to get back to Q16.16 for dividing by 2 etc.
    // term1 = 1.0 (Q16.16)
    wire signed [`QWIDTH-1:0] one = `QONE;

    // -y (already Q16.16)
    wire signed [`QWIDTH-1:0] term_y = -y_in;

    // + y^2 / 2  -> y2 is Q32.32, shift right 16 to Q16.16, then divide by 2 = >>1
    wire signed [`QWIDTH-1:0] term_y2 = $signed(y2 >>> `Q) >>> 1;

    // - y^3 / 6  -> y3 is Q48.48, shift right 32 to get Q16.16, then divide by 6 (approx multiply by reciprocal)
    wire signed [`QWIDTH-1:0] term_y3 = -($signed(y3 >>> (2*`Q)) / 6);

    assign y_out = one + term_y + term_y2 + term_y3;
endmodule
