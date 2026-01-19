#ifndef LMP_RANN_FINGERPRINT_RADIAL_H
#define LMP_RANN_FINGERPRINT_RADIAL_H

#include "rann_fingerprint.h"

namespace LAMMPS_NS {
namespace RANN {

  class Fingerprint_radial : public Fingerprint {
   public:
    Fingerprint_radial(PairRANN *);
    ~Fingerprint_radial();
    bool parse_values(std::string, std::vector<std::string>);
    void write_values(FILE *);
    void init(int *, int);
    void allocate();
    void compute_fingerprint(double *, double *, double *, double *, int, int, double *, double *,
                             double *, int *, int, int *);
    int get_length();

    double *radialtable;
    double *dfctable;
    double dr;
    double *alpha;
    double re;
    int nmax;    //highest term
    int omin;    //lowest term
  };

}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* FINGERPRINT_RADIAL_H_ */
