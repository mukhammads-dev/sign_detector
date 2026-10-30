import type { Metadata } from "next";
import "./globals.scss";

export const metadata: Metadata = {
  title: "SignGuard AI",
  description:
    "Traffic sign recognition that returns UNKNOWN instead of guessing.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
