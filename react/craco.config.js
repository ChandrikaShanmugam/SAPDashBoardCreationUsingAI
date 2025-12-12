module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Find the source-map-loader rule and exclude plotly.js
      webpackConfig.module.rules.forEach((rule) => {
        if (rule.loader && rule.loader.includes('source-map-loader')) {
          // Create exclude array if it doesn't exist
          if (!rule.exclude) {
            rule.exclude = [];
          } else if (!Array.isArray(rule.exclude)) {
            // If exclude is not an array, convert it to an array
            rule.exclude = [rule.exclude];
          }
          // Exclude plotly.js from source map processing
          rule.exclude.push(/node_modules\/plotly\.js/);
        }
      });

      return webpackConfig;
    },
  },
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
};