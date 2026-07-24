# Third-party notices

This repository redistributes the following third-party components. They are
**not** covered by the repository's LICENSE and remain governed by their own
terms, reproduced below.

---

## Cytoscape.js — `frontend/vendor/cytoscape.min.js`

Version 3.30.4 · https://js.cytoscape.org · **MIT License**

> Copyright (c) 2016-2024, The Cytoscape Consortium.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Cytoscape.js bundles `es6-promise` by Ralf S. Engelschall (MIT), whose notice is
retained in the distributed file header.

---

## Apache ECharts — `frontend/vendor/echarts.min.js`

Version 5.5.1 · https://echarts.apache.org · **Apache License, Version 2.0**

> Copyright The Apache Software Foundation.
>
> Licensed to the Apache Software Foundation (ASF) under one or more
> contributor license agreements. See the NOTICE file distributed with this
> work for additional information regarding copyright ownership. The ASF
> licenses this file to you under the Apache License, Version 2.0 (the
> "License"); you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

The file is redistributed unmodified, with its original license header intact.

---

## Runtime dependencies (not redistributed)

Installed from PyPI at setup time via `requirements.txt`; each remains under its
own license. Listed for transparency only:

| Package | License |
|---|---|
| NumPy, SciPy, pandas, NetworkX | BSD-3-Clause |
| PyTorch | BSD-style (see pytorch.org) |
| Gymnasium, Stable-Baselines3, sb3-contrib | MIT |
| FastAPI, uvicorn, websockets, httpx, pydantic | MIT / BSD-3-Clause |
| Matplotlib | PSF-based (matplotlib license) |
| PyYAML | MIT |
| TensorBoard | Apache-2.0 |
| pytest | MIT |
