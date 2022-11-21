# DynamicESF

Author implementation of DynamicESF model, a computationally efficient spatially and temporally varying coefficient model.
DynamicESF extends SVC (Spatially Varying Coefficient) models for space-time analysis.

## Install

```
pip install DynamicESF
```

See https://pypi.org/project/DynamicESF/ for detail.

## Examples

Check out [jupyter notebooks](https://github.com/hayato-n/DynamicESF/blob/main/examples).

## Reference

Please cite the following article.

- Nishi, H., Asami, Y., Baba, H., & Shimizu, C. (2022). Scalable spatiotemporal regression model based on Moran’s eigenvectors. *International Journal of Geographical Information Science*, 1–27. https://doi.org/10.1080/13658816.2022.2100891

I recommend checking the following paper, which proposed the approximation method of Moran's eigenvectors.

- Murakami, D., & Griffith, D. A. (2019). Eigenvector Spatial Filtering for Large Data Sets: Fixed and Random Effects Approaches. *Geographical Analysis*, 51(1), 23–49. https://doi.org/10.1111/gean.12156

And R package `spmoran` (https://cran.r-project.org/web/packages/spmoran/index.html) will be helpful for R users.
=======
Nishi, H., Asami, Y., Baba, H., & Shimizu, C. (2022). Scalable spatiotemporal regression model based on Moran’s eigenvectors. *International Journal of Geographical Information Science*, 1–27. https://doi.org/10.1080/13658816.2022.2100891
