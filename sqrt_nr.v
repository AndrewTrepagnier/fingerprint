// sqrt_nr.v
`include "fixed_pkg.v"
module sqrt_nr #(
    parameter ITER = 3
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [63:0] r2_in,    // Q32.32
    output reg  [31:0] r_out,    // Q16.16
    output reg  valid
);
    // We'll perform NR on fixed point: sqrt(x). Convert to Q16.16 at output.
    // Use an iterative integer method: x_{n+1} = 0.5*(x_n + N/x_n)
    // Work internally in Q16.16 or Q32.32 as needed.

    reg [63:0] N;
    reg [31:0] xk; // current estimate Q16.16
    integer iter;
    reg busy;

    function [31:0] initial_guess;
        input [63:0] val;
        // crude initial guess: shift right by half the integer bits
        begin
            if (val == 0) initial_guess = 0;
            else begin
                // Take msb position to approximate sqrt
                integer msb; integer tmp;
                tmp = val;
                msb = 0;
                while (tmp != 0) begin
                    tmp = tmp >> 1;
                    msb = msb + 1;
                end
                // make initial guess: 1 << ((msb/2)-16) scaled to Q16.16
                integer gexp;
                gexp = (msb/2) - 16;
                if (gexp < 0) initial_guess = (1 << 16) >> (-gexp);
                else initial_guess = (1 << 16) << gexp;
            end
        end
    endfunction

    // Integer division helper (simple non-pipelined) for N/xk
    function [31:0] div_q; // returns Q16.16
        input [63:0] numer; // Q32.32
        input [31:0] denom; // Q16.16
        reg [95:0] numer_sh;
        begin
            if (denom == 0) div_q = 32'h7fffffff;
            else begin
                numer_sh = {numer, 32'b0}; // shift left 32 to divide, now Q64.64
                div_q = numer_sh / denom;  // produce Q48.48 then truncated -> we take upper bits
            end
        end
    endfunction

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            busy <= 0;
            valid <= 0;
            r_out <= 0;
        end else begin
            if (start && !busy) begin
                N <= r2_in;
                xk <= initial_guess(r2_in);
                iter <= 0;
                busy <= 1;
                valid <= 0;
            end else if (busy) begin
                if (iter < ITER) begin
                    // compute x_{n+1} = 0.5*(xk + N/xk)
                    // N/xk in Q16.16 via div helper
                    reg [31:0] nx;
                    nx = div_q(N, xk); // Q16.16
                    // average: (xk + nx)/2
                    xk <= (xk + nx) >>> 1;
                    iter <= iter + 1;
                end else begin
                    r_out <= xk;
                    valid <= 1;
                    busy <= 0;
                end
            end else begin
                valid <= 0; // clear valid until next start
            end
        end
    end
endmodule
