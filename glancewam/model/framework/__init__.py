"""Framework package internals.

Public framework APIs are defined in `glancewam.model.framework.base_framework`.
The `FRAMEWORK_REGISTRY` lives in `glancewam.model.tools` — import it from there
directly. We avoid eager imports here so that lightweight modules under this
package (e.g. `share_tools`) can be loaded by torch-free clients.
"""
