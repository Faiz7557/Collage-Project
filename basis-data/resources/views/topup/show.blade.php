@extends('layouts.main')

@section('content')
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">Detail Top-up</div>

                <div class="card-body">
                    <div class="alert alert-info">
                        <p>Silakan lakukan pembayaran dengan detail berikut:</p>
                    </div>

                    <div class="mb-3">
                        <h5>Kode Referensi:</h5>
                        <div class="p-3 bg-light text-center">
                            <h3>{{ $topUp->reference_code }}</h3>
                        </div>
                        <small class="text-muted">Gunakan kode referensi ini saat melakukan pembayaran</small>
                    </div>

                    <div class="mb-3">
                        <h5>Jumlah:</h5>
                        <p class="lead">Rp {{ number_format($topUp->amount, 0, ',', '.') }}</p>
                    </div>

                    <div class="mb-3">
                        <h5>Metode Pembayaran:</h5>
                        <p>{{ $topUp->payment_method == 'bank' ? 'Bank Transfer' : 'Mini Market' }}</p>
                    </div>

                    <div class="mb-3">
                        <h5>Status:</h5>
                        <p>
                            @if($topUp->status == 'pending')
                                <span class="badge bg-warning">Menunggu Pembayaran</span>
                            @elseif($topUp->status == 'completed')
                                <span class="badge bg-success">Selesai</span>
                            @else
                                <span class="badge bg-danger">Gagal</span>
                            @endif
                        </p>
                    </div>

                    @if($topUp->status == 'pending')
                        <form action="{{ route('topup.confirm', $topUp->id) }}" method="POST">
                            @csrf
                            <button type="submit" class="btn btn-success">Sudah Melakukan Pembayaran</button>
                        </form>
                    @endif

                    <div class="mt-3">
                        <a href="{{ route('topup.index') }}" class="btn btn-secondary">Kembali ke Daftar Top-up</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection