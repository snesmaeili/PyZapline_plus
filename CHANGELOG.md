# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with >90% coverage
- Professional documentation with MkDocs
- GitHub Actions for CI/CD and automated publishing
- Type hints throughout the codebase
- Integration examples for MNE-Python

### Changed
- Restructured package for professional distribution
- Improved error handling and validation
- Enhanced visualization capabilities
- Optimized performance for large datasets

### Fixed
- Memory efficiency improvements
- Edge cases in noise detection
- Compatibility with latest NumPy and SciPy versions

## [1.0.0] - 2024-01-XX

### Added
- Initial release of PyZaplinePlus
- Core functionality for line noise removal using Zapline-plus algorithm
- Automatic and manual noise frequency detection
- Adaptive and fixed chunk segmentation
- Comprehensive visualization of cleaning results
- Integration with MNE-Python workflows
- Support for Python 3.8-3.12
- MIT license

### Features
- **Automatic Line Noise Detection**: Detects power line noise (50/60 Hz) automatically
- **Adaptive Processing**: Dynamically adjusts cleaning parameters
- **Denoising Source Separation**: Uses DSS with PCA for precise noise removal  
- **Multiple Noise Frequencies**: Handles multiple frequencies simultaneously
- **Flexible Segmentation**: Both fixed-time and adaptive chunking options
- **Visualization Tools**: Built-in plotting for quality assessment
- **Scientific Validation**: Based on peer-reviewed Zapline-plus research

### Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- Optional: mne >= 1.0.0 for EEG integration

---

## Version History

- **v1.0.0**: Initial professional release with full feature set
- **v0.x.x**: Development versions (not publicly released)

## Migration Guide

### From Original pyZapline_plus

If you were using the original development version:

```python
# Old way
from pyZapline_plus.pyzaplineplus import PyZaplinePlus, zapline_plus

# New way  
from pyzaplineplus import PyZaplinePlus, zapline_plus
```

The API remains largely the same, but with improved:
- Error handling
- Documentation
- Type safety
- Performance
- Testing coverage

## Breaking Changes

- Package name changed from `pyZapline_plus` to `pyzaplineplus`
- Module imports simplified
- Some internal function signatures improved for consistency

## Support

For questions about specific versions or migration help:
- Open an issue on [GitHub](https://github.com/snesmaeili/PyZapline_plus/issues)
- Check the [documentation](https://snesmaeili.github.io/PyZapline_plus/)
- Email: sina.esmaeili@umontreal.ca
