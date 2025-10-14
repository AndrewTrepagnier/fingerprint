// cutoff.v
`include "fixed_pkg.v"
module cutoff (
    input  wire signed [`QWIDTH-1:0] rc,   // Q16.16
    input  wire signed [`QWIDTH-1:0] dr,   // Q16.16
    input  wire signed [`QWIDTH-1:0] r,    // Q16.16
    output wire signed [`QWIDTH-1:0] fc    // Q16.16
);
    // x = (rc - r)/dr
    wire signed [`QWIDTH-1:0] numer = rc - r;
    // divide numer/dr -> domain Q16.16
    // simple division for simulation; replace with IP for synthesis
    wire signed [`QWIDTH-1:0] x;
    assign x = (dr == 0) ? 32'sd0 : (numer * `QONE) / dr; // numerator scaled to Q32 then divided to Q16

    // clamp
    wire signed [`QWIDTH-1:0] x_clamped = (x >= `QONE) ? `QONE : ((x <= 0) ? 0 : x);

    // compute (1 - (1-x)^4)^2
    // t = 1-x_clamped
    wire signed [`QWIDTH-1:0] t = `QONE - x_clamped;

    // t^2 (Q32.32)
    wire signed [63:0] t2 = $signed(t) * $signed(t);
    // t^4 = (t2 >> Q) * (t2 >> Q) -> take t2 as Q32 and shift to Q16 before multiply
    wire signed [63:0] t2_q16 = t2 >>> `Q; // now Q16.16
    wire signed [63:0] t4 = t2_q16 * t2_q16; // Q32.32

    // (1 - t4) -> need to shift t4 to Q16.16
    wire signed [`QWIDTH-1:0] t4_q16 = $signed(t4 >>> `Q);
    wire signed [`QWIDTH-1:0] one_minus_t4 = `QONE - t4_q16;

    // finally square it: (one_minus_t4)^2 -> Q32.32 then take Q16.16
    wire signed [63:0] fc_full = $signed(one_minus_t4) * $signed(one_minus_t4);
    assign fc = $signed(fc_full >>> `Q);

endmodule
