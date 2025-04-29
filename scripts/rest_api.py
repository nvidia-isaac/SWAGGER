# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import uvicorn

import swagger.endpoints as endpoints


def parse_arguments():
    parser = argparse.ArgumentParser(description="The SWAGGER REST API service.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the service to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the service to.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Log level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return parser.parse_args()


def run(host: str, port: int, log_level: str):
    versioned_app = endpoints.get_versioned_app()
    uvicorn.run(versioned_app, host=host, port=port, log_level=log_level)


def main():
    args = parse_arguments()
    run(args.host, args.port, args.log_level)


if __name__ == "__main__":
    main()
