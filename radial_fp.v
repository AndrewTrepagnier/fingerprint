// ============================================================================
// Radial Fingerprint Hardware Unit (MVP)
// ----------------------------------------------------------------------------
// Simplified model of the radial fingerprint computation in hardware.
// Computes pairwise distances between atoms and applies the exponential term.

module radial_fingerprint #(
    parameter NUM_ATOMS = 8,
    parameter DATA_WIDTH = 32
)(
    input wire clk,
    input wire rst,
    input wire start,
    input wire [DATA_WIDTH-1:0] re,      // equilibrium radius
    input wire [DATA_WIDTH-1:0] rc,      // cutoff radius
    input wire [DATA_WIDTH-1:0] alpha,   // exponential decay constant
    input wire [DATA_WIDTH-1:0] dr,      // radial step
    input wire [DATA_WIDTH-1:0] x [0:NUM_ATOMS-1],
    input wire [DATA_WIDTH-1:0] y [0:NUM_ATOMS-1],
    input wire [DATA_WIDTH-1:0] z [0:NUM_ATOMS-1],
    output reg done,
    output reg [DATA_WIDTH-1:0] fingerprint [0:NUM_ATOMS-1]
);

    // Internal registers
    integer i, j;
    real dx, dy, dz;
    real rij, cutoff, radial_term;
    real tmp_sum [0:NUM_ATOMS-1];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
            for (i = 0; i < NUM_ATOMS; i = i + 1)
                fingerprint[i] <= 0;
        end
        else if (start) begin
            // Initialize sums
            for (i = 0; i < NUM_ATOMS; i = i + 1)
                tmp_sum[i] = 0;

            // Nested loop over atoms
            for (i = 0; i < NUM_ATOMS; i = i + 1) begin
                for (j = 0; j < NUM_ATOMS; j = j + 1) begin
                    if (i != j) begin
                        dx = $bitstoreal(x[i]) - $bitstoreal(x[j]);
                        dy = $bitstoreal(y[i]) - $bitstoreal(y[j]);
                        dz = $bitstoreal(z[i]) - $bitstoreal(z[j]);
                        rij = sqrt(dx*dx + dy*dy + dz*dz);

                        // Smooth cutoff function
                        if (rij < rc)
                            cutoff = pow((1 - (rij/rc)), 4);
                        else
                            cutoff = 0;

                        // Radial term ~ (r/re)^m * exp(-alpha * (r/re))
                        radial_term = pow((rij / re), 2.0) * exp(-alpha * (rij / re)) * cutoff;

                        tmp_sum[i] = tmp_sum[i] + radial_term;
                    end
                end
            end

            // Assign computed fingerprints
            for (i = 0; i < NUM_ATOMS; i = i + 1)
                fingerprint[i] <= $realtobits(tmp_sum[i]);

            done <= 1;
        end
    end
endmodule

