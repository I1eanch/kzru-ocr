<?php

declare(strict_types=1);

namespace App\Libraries;

use RuntimeException;

/**
 * Клиент локального OCR-сервиса (kzru-ocr).
 *
 * Вся логика распознавания живёт в сервисе; здесь только транспорт, ретраи и
 * разбор контракта. Сервис слушает на loopback, поэтому ни аутентификации, ни
 * TLS не предполагается — вынос порта наружу запрещён конфигурацией сети.
 *
 * Использование в контроллере CodeIgniter 4:
 *
 *     $client = new OcrClient();
 *     $result = $client->recognize($pathToPdf);
 *     $text   = $result['text'];
 *     foreach ($result['warnings'] as $w) {
 *         // $w['type'] === 'bin_checksum_failed' → показать «проверьте вручную»
 *     }
 */
final class OcrClient
{
    private string $baseUrl;
    private int $timeout;
    private int $connectTimeout;
    private int $retries;

    public function __construct(
        ?string $baseUrl = null,
        int $timeout = 180,
        int $connectTimeout = 3,
        int $retries = 2
    ) {
        $this->baseUrl = rtrim($baseUrl ?? (getenv('OCR_SERVICE_URL') ?: 'http://127.0.0.1:8099'), '/');
        $this->timeout = $timeout;
        $this->connectTimeout = $connectTimeout;
        $this->retries = $retries;
    }

    /**
     * Синхронное распознавание. Профили: fast | balanced | accurate.
     *
     * @return array{text:string,pages:array,fields:array,warnings:array,elapsed_s:float,mean_conf:float}
     */
    public function recognize(string $pdfPath, string $profile = 'balanced', string $render = 'default'): array
    {
        if (!is_readable($pdfPath)) {
            throw new RuntimeException("PDF недоступен для чтения: {$pdfPath}");
        }

        $url = $this->baseUrl . '/ocr?profile=' . urlencode($profile) . '&render=' . urlencode($render);

        $lastError = '';
        for ($attempt = 0; $attempt <= $this->retries; $attempt++) {
            try {
                return $this->post($url, $pdfPath);
            } catch (RuntimeException $e) {
                $lastError = $e->getMessage();
                // 4xx — ошибка запроса, повтор бессмыслен.
                if (str_starts_with($lastError, 'HTTP 4')) {
                    throw $e;
                }
                if ($attempt < $this->retries) {
                    usleep(250_000 * (int) pow(2, $attempt));
                }
            }
        }

        throw new RuntimeException("OCR-сервис недоступен после повторов: {$lastError}");
    }

    /** Проверка живости сервиса и состава движков. */
    public function health(): array
    {
        $ch = curl_init($this->baseUrl . '/healthz');
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 5,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
        ]);
        $body = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($body === false) {
            throw new RuntimeException("healthz недоступен: {$err}");
        }
        if ($code !== 200) {
            throw new RuntimeException("HTTP {$code} на healthz");
        }

        return $this->decode((string) $body);
    }

    private function post(string $url, string $pdfPath): array
    {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST           => true,
            CURLOPT_TIMEOUT        => $this->timeout,
            CURLOPT_CONNECTTIMEOUT => $this->connectTimeout,
            CURLOPT_POSTFIELDS     => [
                'file' => new \CURLFile($pdfPath, 'application/pdf', basename($pdfPath)),
            ],
        ]);

        $body = curl_exec($ch);
        $code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $err  = curl_error($ch);
        curl_close($ch);

        if ($body === false) {
            throw new RuntimeException("curl: {$err}");
        }
        if ($code !== 200) {
            $detail = '';
            $decoded = json_decode((string) $body, true);
            if (is_array($decoded) && isset($decoded['detail'])) {
                $detail = is_string($decoded['detail']) ? $decoded['detail'] : json_encode($decoded['detail']);
            }
            throw new RuntimeException("HTTP {$code}: {$detail}");
        }

        return $this->decode((string) $body);
    }

    private function decode(string $body): array
    {
        $data = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
        if (!is_array($data)) {
            throw new RuntimeException('сервис вернул не JSON-объект');
        }

        return $data;
    }
}
