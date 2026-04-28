# WebView 安卓电视 APK 打包说明

本项目用于将 https://www.fczhibo.net/live/nba/index.html 打包为安卓 APK，可在安卓电视上安装。

## 目录结构
- app/src/main/java/com/example/webviewapp/MainActivity.java  // 主入口，WebView 加载网页
- app/src/main/AndroidManifest.xml  // 权限与应用声明
- app/build.gradle  // App 构建配置
- build.gradle  // 项目构建配置

## 打包步骤

1. 安装 Android Studio（或命令行构建工具）。
2. 用 Android Studio 打开 pack_apk 目录。
3. 连接安卓电视或模拟器，或生成 APK 文件：
   - 运行 `Build > Build Bundle(s) / APK(s) > Build APK(s)`
   - 生成的 APK 位于 `app/build/outputs/apk/`
4. 拷贝 APK 到电视，允许“未知来源”后安装。

## 主要代码说明

**MainActivity.java**
```java
WebView webView = new WebView(this);
webView.getSettings().setJavaScriptEnabled(true);
webView.setWebViewClient(new WebViewClient());
webView.loadUrl("https://www.fczhibo.net/live/nba/index.html");
setContentView(webView);
```

**AndroidManifest.xml**
```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

## 注意事项
- 电视需联网。
- 体验取决于电视 WebView 兼容性。
- 如需自定义网址或界面，可修改 MainActivity.java。
