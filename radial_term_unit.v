// radial_term_unit.v
`include "fixed_pkg.v"
module radial_term_unit #(
    parameter M = 1  // m value
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire signed [`QWIDTH-1:0] re,    // Q16.16
    input  wire signed [`QWIDTH-1:0] alpha, // Q16.16
    input  wire signed [`QWIDTH-1:0] r,     // Q16.16
    input  wire signed [`QWIDTH-1:0] rc,
    input  wire signed [`QWIDTH-1:0] dr,
    output reg  signed [`QWIDTH-1:0] term,  // Q16.16
    output reg valid
);

    // compute u = r/re  (Q16.16)  => use scaling multiply then divide
    wire signed [`QWIDTH-1:0] u;
    assign u = (re == 0) ? 0 : (r * `QONE) / re;

    // compute u^m by repeated multiply (for m=0 => 1)
    function [31:0] pow_um;
        input integer mm;
        input [31:0] uu;
        integer idx;
        reg [63:0] acc;
        begin
            if (mm == 0) pow_um = `QONE;
            else begin
                acc = uu;
                for (idx = 1; idx < mm; idx = idx + 1) begin
                    acc = (acc * uu) >>> `Q; // keep Q16.16
                end
                pow_um = acc[31:0];
            end
        end
    endfunction

    // instantiate cutoff and exp_approx
    wire signed [`QWIDTH-1:0] fc;
    wire signed [`QWIDTH-1:0] y; // y = alpha * u
    wire signed [`QWIDTH-1:0] expy;

    assign y = ($signed(alpha) * $signed(u)) >>> `Q; // alpha * (r/re)

    exp_approx exp0 (.y_in(y), .y_out(expy));
    cutoff cut0 (.rc(rc), .dr(dr), .r(r), .fc(fc));

    reg [31:0] u_pow;
    reg [31:0] mul_tmp;

    always @(posedge clk) begin
        if (rst) begin
            term <= 0;
            valid <= 0;
        end else if (start) begin
            u_pow <= pow_um(M, u); // Q16.16
            // multiply u_pow (Q) * expy (Q) -> Q32 then shift
            mul_tmp <= ($signed(u_pow) * $signed(expy)) >>> `Q; // Q16.16
            // multiply by fc
            term <= ($signed(mul_tmp) * $signed(fc)) >>> `Q;
            valid <= 1;
        end else begin
            valid <= 0;
        end
    end

endmodule
