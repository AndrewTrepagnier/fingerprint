// dist_sq.v
`include "fixed_pkg.v"
module dist_sq (
    input  wire signed [`QWIDTH-1:0] xi,
    input  wire signed [`QWIDTH-1:0] yi,
    input  wire signed [`QWIDTH-1:0] zi,
    input  wire signed [`QWIDTH-1:0] xj,
    input  wire signed [`QWIDTH-1:0] yj,
    input  wire signed [`QWIDTH-1:0] zj,
    output wire signed [63:0] r2   // full 64-bit square sum in Q32.32 (before normalization)
);

    // dx,dy,dz in Q16.16
    wire signed [`QWIDTH-1:0] dx = xi - xj;
    wire signed [`QWIDTH-1:0] dy = yi - yj;
    wire signed [`QWIDTH-1:0] dz = zi - zj;

    // squares: each multiplication yields Q32.32-like. We'll keep full width.
    wire signed [63:0] dx2 = $signed(dx) * $signed(dx);
    wire signed [63:0] dy2 = $signed(dy) * $signed(dy);
    wire signed [63:0] dz2 = $signed(dz) * $signed(dz);

    assign r2 = dx2 + dy2 + dz2; // still in (Q16.16 * Q16.16) => Q32.32 format

endmodule
