{
  description = "Python env for K-means TP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python313
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];
      };
    };
}
